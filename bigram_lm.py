#! /usr/bin/env python

from pathlib import Path
import torch.nn as nn
import torch
from torch.nn import functional as fn

torch.manual_seed(1337)
text_path = Path("./input.txt")

block_size = 8
batch_size = 32
eval_iters = 100
eval_interval = 1000
max_iters = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


text = text_path.read_text()
voc = sorted(set(text))
voc_len = len(voc)


i2s = dict(enumerate(voc))
s2i = {v: k for (k, v) in i2s.items()}


def encode(s: str) -> list[int]:
    return [s2i[k] for k in s]


def decode(i: list[int]) -> str:
    return "".join([i2s[k] for k in i])


data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data) * 0.9)
train_data = data[:split]
val_data = data[split:]


def get_batch(is_train: bool = True):
    data = train_data if is_train else val_data
    ids = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ids])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ids])
    x, y = x.to(device), y.to(device)
    return x, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.reshape(B * T, C)
        targets = targets.reshape(B * T)

        loss = fn.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            probs = fn.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, 1)  # (batch_size, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # (batch_size, T + 1)

        return idx


m = BigramLanguageModel(voc_len)
m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for is_train in [True, False]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(is_train)
            _, loss = m(x, y)
            losses[i] = loss.item()
        out["train_loss" if is_train else "val_loss"] = losses.mean()
    m.train()
    return out


for step in range(max_iters):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        out = estimate_loss()
        print(
            f"step {step}: train_loss={out['train_loss']:.4f}, val_loss={out['val_loss']:.4f}"
        )


print(decode(m.generate(torch.tensor([[0]]).to(device)).tolist()[0]))
