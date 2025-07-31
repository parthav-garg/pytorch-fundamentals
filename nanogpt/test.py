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