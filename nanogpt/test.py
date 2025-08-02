import torch
import torch.nn as nn
import torch.nn.functional as F
from nanogpt.train import GPT, decode, encode, device
import os

model = GPT()
if os.path.exists("savez/model.pth"):
    print("Loading existing model...")
    model.load_state_dict(torch.load("savez/model.pth"))
print("Model loaded.")
model.to(device)
model.eval()

with torch.no_grad():
    with open("nanogpt/output.txt", "r") as f:
        content = f.read()
        context = torch.tensor(encode(content[-model.block_size:]), dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(context, max_new_tokens=1000)[0].tolist()
        print("Generated text:")
        print(decode(generated))



class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv_kernel: int = 4, expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model # D
        self.d_state = d_state # N
        self.d_conv_kernel = d_conv_kernel
        self.expand_factor = expand_factor

        self.d_inner = self.d_model * self.expand_factor # D_inner

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv_kernel,
            padding=(d_conv_kernel - 1) // 2,
            groups=self.d_inner,
            bias=False
        )

        self.conv_act = nn.SiLU()

        self.s_B_linear = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.s_C_linear = nn.Linear(self.d_inner, self.d_state, bias=False)

        self.delta_base_param = nn.Parameter(torch.randn(self.d_model)) 
        self.s_delta_linear = nn.Linear(self.d_inner, self.d_model, bias=False) 
        self.tau_delta_activation = nn.Softplus()
        self.tau_delta_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1)).repeat(self.d_model, 1))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)


    @staticmethod
    def _discretize_mamba(A_log, B_proj, Delta, d_inner, d_state):
        # A_log: (D_inner, D_state) -> used to create A (1, 1, D_inner, D_state)
        # B_proj: (B, L, D_inner, D_state) <- NEW SHAPE HERE
        # Delta: (B, L, D_inner) -> used to create Delta_exp (B, L, D_inner, 1)

        A = -torch.exp(A_log).unsqueeze(0).unsqueeze(0) # (1, 1, D_inner, D_state)

        Delta_exp = Delta.unsqueeze(-1) # (B, L, D_inner, 1)

        # B_exp is now simply B_proj, no unsqueeze needed here because it's already (B, L, D_inner, D_state)
        B_exp = B_proj # Shape (B, L, D_inner, D_state)

        delta_A = Delta_exp * A # Result (B, L, D_inner, D_state)
        A_bar = torch.exp(delta_A)

        B_bar_frac = torch.special.expm1(delta_A) / (delta_A + 1e-6)
        B_bar = B_bar_frac * Delta_exp * B_exp # All terms are (B,L,D_inner,D_state) or (B,L,D_inner,1), broadcasts correctly
        return A_bar, B_bar

    @staticmethod
    def _ssm_scan(A_bar, B_bar, C_proj, u):
        B, L, D_inner, D_state = A_bar.shape
        h = torch.zeros(B, D_inner, D_state, device=A_bar.device)
        ys = []

        for t in range(L):
            A_bar_t = A_bar[:, t]       # (B, D_inner, D_state)
            B_bar_t = B_bar[:, t]       # (B, D_inner, D_state)
            C_t = C_proj[:, t]          # (B, D_inner, D_state) <- NEW SHAPE HERE
            u_t = u[:, t]               # (B, D_inner)

            # State update: h = (B, D_inner, D_state)
            h = A_bar_t * h + B_bar_t * u_t.unsqueeze(-1)

            # Output calculation: h (B, D_inner, D_state) * C_t (B, D_inner, D_state)
            # The multiplication is element-wise now.
            # Then sum across the D_state dimension to get (B, D_inner)
            y_t = (h * C_t).sum(dim=-1) # y_t (B, D_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # y (B, L, D_inner)
        return y


    def forward(self, x: torch.Tensor):
        # x shape: (B, L, D_model)
        B, L, D_model = x.shape # D_model is self.d_model

        # --- Residual 1: Outer residual for the whole block ---
        residual = x # Store the original input for final addition

        # 1. Initial Linear Projection and Gated Split
        # z: (B, L, D_inner * 2)
        z = self.in_proj(x)
        print(f"z shape after in_proj: {z.shape}") # Debugging line
        # Split into `val` (goes through conv + SSM) and `gate` (for final output gating)
        # val: (B, L, D_inner), gate: (B, L, D_inner)
        val, gate = z.chunk(2, dim=-1)
        print(f"val shape after split: {val.shape}, gate shape: {gate.shape}") # Debugging line
        # 2. Convolutional Block (Applied to `val` branch)
        # Permute for Conv1D: (B, L, D_inner) -> (B, D_inner, L)
        val = val.transpose(-1, -2)
        val = self.conv1d(val)
        val = self.conv_act(val) # Apply activation after convolution
        # Permute back: (B, D_inner, L) -> (B, L, D_inner)
        val = val.transpose(1, 2)
        print(f"val shape after conv: {val.shape}") # Debugging line
        # `val` is now the input `u` for the SSM scan, processed by conv.


        # 3. Selection Mechanism for B, C, Delta (using `val` as input)
        # B_proj: (B, L, D_state) - `s_B_linear` maps D_inner -> D_state
        B_proj = self.s_B_linear(val)
        # C_proj: (B, L, D_state) - `s_C_linear` maps D_inner -> D_state
        C_proj = self.s_C_linear(val)
        
        # Delta calculation:
        # a. Dynamic part from `val` (D_inner -> D_inner)
        delta_from_val = self.s_delta_linear(val) # (B, L, D_inner)
        # b. Add base parameter (D_inner) with broadcasting
        combined_delta = delta_from_val + self.delta_base_param
        print(f"combined_delta shape: {combined_delta.shape}") # Debugging line
        # c. Project and apply Softplus for positivity
        Delta = self.tau_delta_activation(self.tau_delta_projection(combined_delta))


        # 4. Discretization (A_bar, B_bar)
        # A_log is (D_inner, D_state) parameter
        # B_proj (B, L, D_state), Delta (B, L, D_inner)
        print(f"A_log shape: {self.A_log.shape}, B_proj shape: {B_proj.shape}, Delta shape: {Delta.shape}") # Debugging line
        A_bar, B_bar = self._discretize_mamba(self.A_log, B_proj, Delta, self.d_inner, self.d_state)
        # A_bar, B_bar shapes: (B, L, D_inner, D_state)
        print("A_bar shape:", A_bar.shape, "B_bar shape:", B_bar.shape) # Debugging line

        # 5. SSM Scan (Single scan operation)
        # ssm_output: (B, L, D_inner)
        ssm_output = self._ssm_scan(A_bar, B_bar, C_proj, val)


        # 6. Final Gating
        # gate_activated: (B, L, D_inner) - from earlier split, applies activation
        gate_activated = F.silu(gate) # Apply SiLU activation to the gate branch
        
        # Multiply SSM output with activated gate
        # gated_output: (B, L, D_inner)
        gated_output = ssm_output * gate_activated
        
        # 7. Final Projection from D_inner back to D_model
        # final_block_output: (B, L, D_model)
        final_block_output = self.out_proj(gated_output)


        # --- Residual 2: Add original input back ---
        # output: (B, L, D_model)
        output = final_block_output + residual

        return output
 