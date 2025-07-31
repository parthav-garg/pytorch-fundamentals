import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import os
content = ""
with open("nanogpt/input.txt", "r") as file:
    content = file.read()

vocab = sorted(list(set(content)))
vocab_size = len(vocab)
device = "cuda" if torch.cuda.is_available() else "cpu"
stoi = { ch: i for i, ch in enumerate(vocab) }
itos = { i: ch for i, ch in enumerate(vocab) }
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[i] for i in x])
data = torch.tensor(encode(content), dtype=torch.long)
n = int(len(data) * .9)
train_data = data[:n]
val_data = data[n:]
torch.manual_seed(1337)

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x_data =  torch.stack([data[i : i + block_size] for i in ix])
    y_data =  torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x_data = x_data.to(device)
    y_data = y_data.to(device) 
    return x_data, y_data


def estimate_loss(model, eval_iter):
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iter)
            for i in range(eval_iter):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

criterion = nn.CrossEntropyLoss()


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        similarity = q @ k.transpose(-2, -1) #(B, C , T)
        similarity = similarity * C ** -.5
        similarity = torch.softmax(similarity.masked_fill(self.tril[:T, :T] ==0, float("-inf")), dim = -1)
        similarity = self.dropout(similarity)
        out = similarity @ v
        
        return out
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x =torch.cat([h(x) for h in self.heads], dim = 2)
        x = self.dropout(self.proj(x))
        return x

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_embd, n_embd * 4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(n_embd * 4, n_embd)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.l1, self.relu, self.l2, self.drop)
    def forward(self, x):
        return self.net(x)
        
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FeedForward()
        self.attn = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return self.drop(x)
    
class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = criterion(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):    
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
torch.set_float32_matmul_precision('high')
m = GPT()
def run(m):
    if os.path.exists("savez/model.pth"):
        print("Loading existing model...") 
        m.load_state_dict(torch.load("savez/nanogpt_model.pth"))
    else:
        print("Training new model...")
    m.to(device)
    optim = torch.optim.AdamW(m.parameters(), lr=.001)
    for steps in range(5000):
        xb, yb = get_batch("train")
        with torch.autocast("cuda"):
            logits , loss = m(xb, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (steps % 1000 == 0 and steps > 0):
            print(estimate_loss(m, 10))
    print("Saving model...")
    torch.save(m.state_dict(), "savez/nanogpt_model.pth")
    print("Training complete. Generating text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = m.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(generated))
if __name__ == "__main__":
    run(m)