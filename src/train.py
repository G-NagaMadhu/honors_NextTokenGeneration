"""Train a small decoder-only character-level MiniGPT on a text corpus."""

import argparse
import math
import os
import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2


class Head(nn.Module):
    """One causal self-attention head.

    last_attention is diagnostic-only. It adds no trainable parameters, so
    checkpoints trained before this change remain compatible.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x):
        _, time, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = q @ k.transpose(-2, -1) * channels ** -0.5
        scores = scores.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        self.last_attention = attention.detach()
        return self.dropout(attention) @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)]
        )
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    config.n_embd,
                    config.n_head,
                    config.block_size,
                    config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(
                f"Sequence length {time} exceeds block_size {self.block_size}."
            )
        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            batch, target_time, channels = logits.shape
            loss = F.cross_entropy(
                logits.view(batch * target_time, channels),
                targets.view(batch * target_time),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cumulative_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_indices, sorted_logits)

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path, block_size):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    if len(text) < block_size + 2:
        raise ValueError("Corpus is too small for the chosen block_size.")
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    encode = lambda value: [stoi[char] for char in value]
    decode = lambda ids: "".join(itos[index] for index in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    return text, data[:split], data[split:], encode, decode, len(chars)


def get_batch(split, train_data, val_data, config, device):
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([source[start : start + config.block_size] for start in starts])
    y = torch.stack(
        [source[start + 1 : start + config.block_size + 1] for start in starts]
    )
    return x.to(device), y.to(device)


def estimate_loss(model, train_data, val_data, config, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for split in ("train", "val"):
            losses = torch.zeros(config.eval_iters)
            for index in range(config.eval_iters):
                x, y = get_batch(split, train_data, val_data, config, device)
                _, loss = model(x, y)
                losses[index] = loss.item()
            results[split] = losses.mean().item()
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--checkpoint", default="checkpoints/shakespeare_char.pt")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    config = Config(
        max_iters=args.max_iters,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text, train_data, val_data, encode, decode, vocab_size = load_data(
        args.data, config.block_size
    )
    print(f"Characters: {len(text):,} | Vocabulary: {vocab_size} | Device: {device}")
    print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")

    model = MiniGPT(vocab_size, config).to(device)
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.max_iters):
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"val perplexity {math.exp(min(losses['val'], 20)):.2f}"
            )
        x, y = get_batch("train", train_data, val_data, config, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    stoi = {char: index for index, char in enumerate(sorted(set(text)))}
    itos = {index: char for index, char in enumerate(sorted(set(text)))}
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocab_size": vocab_size,
            "stoi": stoi,
            "itos": itos,
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")

    prompt = "ROMEO:"
    if all(char in stoi for char in prompt):
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=20)
        print("\nSample:\n" + decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
