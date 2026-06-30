"""Decoding / generation script for the CS336 Transformer language model."""

import argparse
import json

import torch

from tests.adapters import Tokenizer, run_transformer_lm


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample from the smallest set of tokens whose cumulative probability >= p."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # remove tokens once cumulative prob exceeds p (shift by 1 to keep the token that pushes over)
    mask = cumulative_probs - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[next_token]


@torch.no_grad()
def generate(
    model_weights: dict,
    model_config: dict,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """
    Generate tokens autoregressively from a prompt.

    Args:
        model_weights: state dict for the transformer.
        model_config: dict with vocab_size, context_length, d_model, num_layers,
                      num_heads, d_ff, rope_theta.
        prompt_ids: list of integer token ids as the initial context.
        max_new_tokens: maximum number of tokens to generate.
        temperature: softmax temperature. <1 = sharper, >1 = flatter. 1.0 = no scaling.
        top_p: nucleus sampling threshold. 1.0 = no filtering.
        eos_token_id: stop generation when this token is sampled.
        device: torch device string.

    Returns:
        List of generated token ids (not including the prompt).
    """
    context_length = model_config["context_length"]
    ids = list(prompt_ids)
    generated = []

    for _ in range(max_new_tokens):
        # truncate to context window
        input_ids = ids[-context_length:]
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        # forward pass → (1, seq_len, vocab_size)
        logits = run_transformer_lm(**model_config, weights=model_weights, in_indices=x)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        probs = torch.softmax(next_logits, dim=-1)

        # top-p (nucleus) sampling
        if top_p < 1.0:
            next_token = sample_top_p(probs, top_p).item()
        else:
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        ids.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer_vocab", required=True, help="Path to tokenizer vocab JSON")
    parser.add_argument("--tokenizer_merges", required=True, help="Path to tokenizer merges JSON")
    parser.add_argument("--prompt", default="", help="Text prompt to complete")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_config = ckpt["model_config"]
    model_weights = {k: v.to(args.device) for k, v in ckpt["model"].items()}

    # Load tokenizer
    with open(args.tokenizer_vocab) as f:
        vocab = {int(k): v.encode("latin-1") for k, v in json.load(f).items()}
    with open(args.tokenizer_merges) as f:
        merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in json.load(f)]
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    eos_token_id = None
    for tok_id, tok_bytes in vocab.items():
        if tok_bytes == b"<|endoftext|>":
            eos_token_id = tok_id
            break

    prompt_ids = tokenizer.encode(args.prompt) if args.prompt else []
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")

    generated_ids = generate(
        model_weights=model_weights,
        model_config=model_config,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device,
    )

    completion = tokenizer.decode(generated_ids)
    print(f"\nCompletion:\n{completion}")


if __name__ == "__main__":
    main()
