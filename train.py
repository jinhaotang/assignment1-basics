"""Training script for the CS336 Transformer language model."""

import argparse
import math
import os
import time

import numpy as np
import torch

from logger import ExperimentLogger
from tests.adapters import (
    AdamW,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_load_checkpoint,
    run_save_checkpoint,
    run_transformer_lm,
)


def build_model_weights(
    vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device
):
    """Initialize random transformer weights matching the expected state dict keys."""
    weights = {}
    std = 0.02

    def param(shape):
        return torch.nn.Parameter(torch.empty(shape, device=device).normal_(0, std))

    weights["token_embeddings.weight"] = param((vocab_size, d_model))
    for i in range(num_layers):
        p = f"layers.{i}"
        weights[f"{p}.attn.q_proj.weight"] = param((d_model, d_model))
        weights[f"{p}.attn.k_proj.weight"] = param((d_model, d_model))
        weights[f"{p}.attn.v_proj.weight"] = param((d_model, d_model))
        weights[f"{p}.attn.output_proj.weight"] = param((d_model, d_model))
        weights[f"{p}.ln1.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device))
        weights[f"{p}.ffn.w1.weight"] = param((d_ff, d_model))
        weights[f"{p}.ffn.w2.weight"] = param((d_model, d_ff))
        weights[f"{p}.ffn.w3.weight"] = param((d_ff, d_model))
        weights[f"{p}.ln2.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device))
    weights["ln_final.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device))
    weights["lm_head.weight"] = param((vocab_size, d_model))
    return weights


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta,
        )
        weights = build_model_weights(
            vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device
        )
        for k, v in weights.items():
            self.register_parameter(k.replace(".", "_"), v)
        self._weight_keys = list(weights.keys())

    def forward(self, x):
        weights = {k: getattr(self, k.replace(".", "_")) for k in self._weight_keys}
        return run_transformer_lm(**self.config, weights=weights, in_indices=x)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, context_length, device, eval_iters=20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = run_get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_data", required=True, help="Path to training .npy file")
    parser.add_argument("--val_data", required=True, help="Path to validation .npy file")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    # Model
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Schedule
    parser.add_argument("--warmup_iters", type=int, default=250)
    parser.add_argument("--max_iters", type=int, default=5000)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    # Logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run_name", default="run", help="Name for this experiment run")
    parser.add_argument("--log_dir", default="logs", help="Directory to write CSV logs")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32

    exp_logger = ExperimentLogger(
        log_dir=args.log_dir,
        run_name=args.run_name,
        config=vars(args),
        use_wandb=args.wandb,
    )

    # Load datasets memory-mapped
    train_data = np.memmap(args.train_data, dtype=dtype, mode="r")
    val_data = np.memmap(args.val_data, dtype=dtype, mode="r")
    assert train_data.max() < args.vocab_size, f"Train token {train_data.max()} >= vocab_size"
    assert val_data.max() < args.vocab_size, f"Val token {val_data.max()} >= vocab_size"
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    model.to(args.device)

    # Compile for faster training (no TF32 on mps)
    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    elif args.device == "cpu":
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = run_load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    model.train()
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        # Update learning rate
        lr = run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = run_get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(x)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
        if total_norm >= args.grad_clip:
            scale = args.grad_clip / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            ms_per_iter = dt * 1000 / args.log_interval
            print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {ms_per_iter:.1f}ms/iter")
            exp_logger.log(it, {"train_loss": loss.item(), "lr": lr, "ms_per_iter": ms_per_iter})
            t0 = time.time()

        if it % args.eval_interval == 0:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.context_length, args.device)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, args.device)
            print(f"iter {it:6d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
            exp_logger.log(it, {"eval_train_loss": train_loss, "eval_val_loss": val_loss})
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{it:07d}.pt")
            run_save_checkpoint(model, optimizer, it, ckpt_path)
            # Patch in model_config so decode.py can load it without re-specifying args
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt["model_config"] = model.config
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    run_save_checkpoint(model, optimizer, args.max_iters, final_path)
    ckpt = torch.load(final_path, map_location="cpu", weights_only=True)
    ckpt["model_config"] = model.config
    torch.save(ckpt, final_path)
    exp_logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
