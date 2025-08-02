import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import math # For A_log initialization
from torch.profiler import profile, record_function, ProfilerActivity
# --- 1. Data Preparation (Tiny Shakespeare) ---
activites = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]
# Define the URL for the Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "input.txt"

# Download the dataset if it doesn't exist
if not os.path.exists(file_path):
    print(f"Downloading {file_path} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file_path} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print("Please ensure you have an internet connection or download it manually.")
        exit()
else:
    print(f"{file_path} already exists. Skipping download.")

# Load the text and build vocabulary
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)} # String to Integer
itos = {i: ch for i, ch in enumerate(chars)} # Integer to String

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the entire text dataset to a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# PyTorch Dataset & DataLoader
from torch.utils.data import Dataset, DataLoader

class TinyShakespeareDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- Model Hyperparameters & Device Setup ---
block_size = 512     
batch_size = 16     
d_model = 64
d_state = 16
d_conv_kernel = 7
expand_factor = 2
n_layer = 2          # Number of Mamba blocks to stack

learning_rate = 1e-3
num_epochs = 1
max_iters = 500     
eval_interval = 200  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Model Parameters: D_model={d_model}, D_state={d_state}, D_inner={d_model*expand_factor}")
print(f"Training Config: Batch={batch_size}, Block={block_size}, Layers={n_layer}, Max_Iters={max_iters}")
print("-" * 50)


dataset = TinyShakespeareDataset(data, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop_last for consistent batch sizes



class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv_kernel: int = 4, expand_factor: int = 2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv_kernel = d_conv_kernel
        self.expand_factor = expand_factor

        self.d_inner = self.d_model * self.expand_factor

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv_kernel,
            padding=d_conv_kernel // 2,
            groups=self.d_inner,
            bias=False
        )
        self.conv_act = nn.SiLU()

   
        self.s_B_linear = nn.Linear(self.d_inner, self.d_inner * self.d_state, bias=False)
        self.s_C_linear = nn.Linear(self.d_inner, self.d_inner * self.d_state, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1)).repeat(self.d_inner, 1))
        self.delta_base_param = nn.Parameter(torch.randn(self.d_inner))
        self.s_delta_linear = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.tau_delta_projection = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.tau_delta_activation = nn.Softplus()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

 
    @staticmethod
    def _discretize_mamba(A_log, B_proj, Delta, d_inner, d_state):
   

        A = -torch.exp(A_log).unsqueeze(0).unsqueeze(0) # (1, 1, D_inner, D_state)

        Delta_exp = Delta.unsqueeze(-1) # (B, L, D_inner, 1)

        B_exp = B_proj # Shape (B, L, D_inner, D_state)

        delta_A = Delta_exp * A
        A_bar = torch.exp(delta_A)

        B_bar_frac = torch.special.expm1(delta_A) / (delta_A + 1e-6)
        B_bar = B_bar_frac * Delta_exp * B_exp y
        return A_bar, B_bar

    @staticmethod
    def _ssm_scan(A_bar, B_bar, C_proj, u):
        B, L, D_inner, D_state = A_bar.shape
        h = torch.zeros(B, D_inner, D_state, device=A_bar.device)
        ys = []

        for t in range(L):
            A_bar_t = A_bar[:, t]       
            B_bar_t = B_bar[:, t]       
            C_t = C_proj[:, t]          
            u_t = u[:, t]               
       
            h = A_bar_t * h + B_bar_t * u_t.unsqueeze(-1)

      
            y_t = (h * C_t).sum(dim=-1) # y_t (B, D_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # y (B, L, D_inner)
        return y


    def forward(self, x: torch.Tensor):
        B, L, D_model = x.shape
        residual = x

        z = self.in_proj(x)
        val, gate = z.chunk(2, dim=-1)

        val = val.transpose(1, 2)
        val = self.conv1d(val)
        val = self.conv_act(val)
        val = val.transpose(1, 2)


        B_proj_raw = self.s_B_linear(val)
        C_proj = self.s_C_linear(val).view(B, L, self.d_inner, self.d_state)
        B_proj = B_proj_raw.view(B, L, self.d_inner, self.d_state)
        delta_from_val = self.s_delta_linear(val)
        combined_delta = delta_from_val + self.delta_base_param
        Delta = self.tau_delta_activation(self.tau_delta_projection(combined_delta))


        A_bar, B_bar = self._discretize_mamba(self.A_log, B_proj, Delta, self.d_inner, self.d_state)



        ssm_output = self._ssm_scan(A_bar, B_bar, C_proj, val)

        gate_activated = F.silu(gate)
        gated_output = ssm_output * gate_activated
        
        final_block_output = self.out_proj(gated_output)
        output = final_block_output + residual

        return output
    
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, block_size: int,
                 d_state: int = 16, d_conv_kernel: int = 4, expand_factor: int = 2):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        self.token_embedding_table = nn.Embedding(vocab_size, d_model)


        self.mamba_blocks = nn.Sequential(
            *[MambaBlock(d_model=d_model, d_state=d_state,
                         d_conv_kernel=d_conv_kernel, expand_factor=expand_factor)
              for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True) 

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, L = idx.shape


        x = self.token_embedding_table(idx)
        x = self.mamba_blocks(x) 


        x = self.ln_f(x) 

        # Language Model Head
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1)) 
            targets_flat = targets.view(-1)              
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
          
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits_last_token = logits[:, -1, :] 


            probs = F.softmax(logits_last_token, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)


            idx = torch.cat((idx, idx_next), dim=1) 
        
        return idx


model = MambaLanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layer=n_layer,
    block_size=block_size,
    d_state=d_state,
    d_conv_kernel=d_conv_kernel,
    expand_factor=expand_factor
)
model.to(device)
if os.path.exists("savez/mamba_model.pth"):
    print("Loading existing model...")
    model.load_state_dict(torch.load("savez/mamba_model.pth"))
else:
    print("No existing model found, starting fresh.")
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Simple Evaluation Function

@torch.no_grad()
def estimate_loss():
    with profile(activities=activites, record_shapes=True, profile_memory=True) as prof:
        with record_function("estimate_loss"):
            out = {}
            model.eval() 
            losses = []
            for batch_idx, (X, Y) in enumerate(dataloader):
                if batch_idx > 50: 
                    break
                X, Y = X.to(device), Y.to(device)
                _, loss = model(X, Y)
                losses.append(loss.item())
            out['train'] = torch.tensor(losses).mean().item()
            model.train() 
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            return out 
    
print(f"Starting training for {max_iters} iterations...")

data_iter = iter(dataloader)
def run(max_iters):
    for iter_num in range(max_iters):

        if iter_num % 10 == 0:
            losses = estimate_loss()
            print(f"Step {iter_num}: Train loss {losses['train']:.4f}")

        # Get a batch of data
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            # If dataloader is exhausted, re-create the iterator
            data_iter = iter(dataloader)
            xb, yb = next(data_iter)
            
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Training complete. Final loss after {max_iters} iterations: {loss.item():.4f}")
    torch.save(model.state_dict(), "savez/mamba_model.pth")
def profile_training():
    sort_by_keyword = "self_" + "cuda" + "_time_total"
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]

    def trace_handler(p):

        print("\n" + "="*20 + f" Profile for Step {p.step_num} " + "="*20)

        key_averages_table = p.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=20 
        )
        print(key_averages_table)
        

        trace_path = f"./tmp/mamba_trace_step_{p.step_num}.json"
        p.export_chrome_trace(trace_path)
        print(f"Chrome trace saved to: {trace_path}")
        print("To view, open Chrome, navigate to 'chrome://tracing', and load the file.")
        print("="*60 + "\n")
    inputs, yb = next(data_iter)
    inputs = inputs.to(device)
    with profile(
        activities=activities,
        on_trace_ready=trace_handler,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True, 
        with_stack=True     

    ) as p:
        for idx in range(5): 
            with record_function("model_inference"):
                model(inputs)
            p.step()


# --- 5. Text Generation ---
def generate_text(model, start_token_id, max_new_tokens=500):
    model.load_state_dict(torch.load("savez/mamba_model.pth"))
    print("\n" + "="*20 + " GENERATING TEXT " + "="*20)

    start_token_id = stoi['\n'] 
    context = torch.tensor([[start_token_id]], device=device) # Shape (1, 1)


    generated_indices = model.generate(context, max_new_tokens=500)

    # Decode and print the generated text
    generated_text = decode(generated_indices[0].tolist())
    print(generated_text)
    print("\n" + "="*20 + " GENERATION COMPLETE " + "="*20)
profile_training()