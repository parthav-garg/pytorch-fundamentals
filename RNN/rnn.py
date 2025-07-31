import torch
import torch.nn as nn

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

n_embd = 128
seq_len = 256
batch_size = len(train_data) // seq_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self , input_size, hidden_size, output_size, seq_len):
        super().__init__()
        self.Wx = nn.Linear(n_embd, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.hidden_state = torch.zeros(batch_size, hidden_size).to(device)
        self.seq_len = seq_len
    def forward(self, x):
        output = []
        for t in range(self.seq_len):
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            h = torch.tanh(self.Wx(x[:, t, :]) + self.Wh(h))
            output.append(self.Wy(h).unsqueeze(1))
        logits = torch.cat(output, dim=1)
        return logits

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = RNN(input_size=n_embd, hidden_size=256, output_size=n_embd, seq_len=seq_len)
        self.encoding = nn.Embedding(vocab_size, n_embd)
    
    def forward(self, x):
        return self.rnn(self.encoding(x))

def get_batch(split, time = 0):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y
def estimate_loss(model, eval_iter):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for x, y in eval_iter:
            logits= model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(eval_iter)
def train(model, epochs=10):
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.to(torch.device("cuda"))
    time = 0
    while time < len(train_data) - seq_len - 1 and time < 250:
        xb, yb = get_batch("train", time)
        logits= model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        time += 1
        if time % 1 == 0:
            print(f"Epoch , Step {time}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "savez/rnn_model.pth")
model = GPT()
criterion = nn.CrossEntropyLoss()
@torch.no_grad()
def generate(model, start_str, max_new_tokens):
    model.eval()
    
    input_ids = torch.tensor([stoi[ch] for ch in start_str], dtype=torch.long).unsqueeze(0).to(device)  # shape (1, len)
    hidden = torch.zeros(1, model.rnn.hidden_size).to(device)

    for _ in range(max_new_tokens):
        # Take only the last token and embed it
        last_token = input_ids[:, -1]  # shape (1,)
        emb = model.encoding(last_token)  # shape (1, emb_dim)

        # Forward one step manually
        h = torch.tanh(model.rnn.Wx(emb) + model.rnn.Wh(hidden))  # (1, hidden_size)
        logits = model.rnn.Wy(h)  # (1, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)  # shape (1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        hidden = h  # update hidden state

    return decode(input_ids[0].tolist())
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Training on GPU")
    train(model)
    print("\nGenerated Text:\n")
    print(generate(model, start_str="T", max_new_tokens=200))