"""Generate text and inspect a trained MiniGPT checkpoint."""

import argparse
import os
import random

import torch

from train import MiniGPT


def load_checkpoint(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    class Config:
        pass

    config = Config()
    for key, value in checkpoint["config"].items():
        setattr(config, key, value)

    model = MiniGPT(checkpoint["vocab_size"], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}
    return model, config, stoi, itos, device


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_prompt(prompt, stoi):
    unknown = sorted(set(char for char in prompt if char not in stoi))
    if unknown:
        raise ValueError(
            f"Prompt contains characters not in the training vocabulary: {unknown}"
        )
    if not prompt:
        raise ValueError("Prompt cannot be empty. Try --prompt \"ROMEO:\".")
    return [stoi[char] for char in prompt]


def decode(ids, itos):
    return "".join(itos[int(token)] for token in ids)


def generate(model, ids, tokens, temperature, top_k, top_p, device):
    context = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
        )
    return output[0].tolist()


def print_model_info(model, config, stoi, device, args, prompt_length):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("MODEL INFORMATION")
    print("-----------------")
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Attention heads: {config.n_head}")
    print(f"Training context length: {config.block_size}")
    print(f"Vocabulary size: {len(stoi)}")
    print(f"Prompt length: {prompt_length}")
    print(f"New tokens requested: {args.tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k if args.top_k > 0 else 'disabled'}")
    print(f"Top-p: {args.top_p if args.top_p is not None else 'disabled'}")


def attention_modules(model):
    """Yield every attention head in layer order."""
    for layer_number, block in enumerate(model.blocks):
        for head_number, head in enumerate(block.attention.heads):
            yield layer_number, head_number, head


def print_attention_summary(model, itos, ids):
    print("\nATTENTION WEIGHTS")
    print("-----------------")
    found = False

    for layer_number, head_number, head in attention_modules(model):
        attention = head.last_attention
        if attention is None:
            continue
        found = True
        # attention shape: batch, sequence, sequence
        values = attention[0, -1, :]
        count = min(5, values.numel())
        strongest = torch.topk(values, k=count).indices.tolist()
        labels = []
        for position in strongest:
            character = itos[int(ids[position])] if position < len(ids) else "?"
            labels.append(f"{position}={character!r} ({values[position].item():.3f})")
        print(f"Layer {layer_number + 1}, head {head_number + 1}: " + ", ".join(labels))

    if not found:
        print("No attention values were captured.")


def save_attention(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for layer_number, head_number, head in attention_modules(model):
        if head.last_attention is not None:
            path = os.path.join(
                output_dir,
                f"layer_{layer_number + 1}_head_{head_number + 1}.pt",
            )
            torch.save(head.last_attention.cpu(), path)
            print(f"Saved attention: {path}")
            saved += 1
    if saved == 0:
        print("No attention files saved.")


def compare_context_lengths(
    model, prompt_ids, lengths, args, config, itos, device
):
    print("\nCONTEXT-LENGTH COMPARISON")
    print("=========================")
    print("This changes how many prompt characters are visible; it does not retrain.")

    for length in lengths:
        if length < 1:
            print(f"Skipping {length}: length must be positive.")
            continue
        actual_length = min(length, config.block_size)
        ids = prompt_ids[-actual_length:]
        set_seed(args.seed)
        generated_ids = generate(
            model,
            ids,
            args.tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            device,
        )
        print(f"\nRequested context: {length}")
        print(f"Used context: {actual_length}")
        print("-" * 30)
        print(decode(generated_ids, itos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/shakespeare_char.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--context_lengths",
        nargs="+",
        type=int,
        default=None,
        help="Compare lengths, for example: --context_lengths 32 64 128",
    )
    parser.add_argument(
        "--save_attention",
        action="store_true",
        help="Save attention tensors under attention_outputs/.",
    )
    parser.add_argument(
        "--attention_dir",
        default="attention_outputs",
        help="Directory used with --save_attention.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")
    if args.top_k < 0:
        parser.error("--top_k cannot be negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top_p must be between 0 and 1")

    set_seed(args.seed)
    model, config, stoi, itos, device = load_checkpoint(args.checkpoint)
    prompt_ids = encode_prompt(args.prompt, stoi)

    print_model_info(model, config, stoi, device, args, len(prompt_ids))
    if len(prompt_ids) > config.block_size:
        print(
            f"Note: only the last {config.block_size} prompt characters "
            "can be used by this model."
        )

    used_prompt_ids = prompt_ids[-config.block_size :]
    generated_ids = generate(
        model,
        used_prompt_ids,
        args.tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        device,
    )

    print("\nGENERATED TEXT")
    print("--------------")
    print(decode(generated_ids, itos))

    final_context_ids = generated_ids[-config.block_size :]
    print_attention_summary(model, itos, final_context_ids)
    if args.save_attention:
        save_attention(model, args.attention_dir)

    if args.context_lengths:
        compare_context_lengths(
            model,
            prompt_ids,
            args.context_lengths,
            args,
            config,
            itos,
            device,
        )


if __name__ == "__main__":
    main()
