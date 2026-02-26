"""
python cs336_basics/train.py \
  --data-path=data/tokens-owt_train.npy \
  --validation-data-path=data/tokens-owt_valid.npy \
  --total-steps=1000 \
  --validation-interval=50 \
  --early-stopping-patience=5 \
  --early-stopping-min-delta=0.001 \
  --checkpoint-dir=./checkpoints
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------------------------
# Configuration
# -------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_path: str
    validation_data_path: Optional[str] = None

    # Model hyperparameters
    vocab_size: int = 50257
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    context_length: int = 128

    # Optimizer hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32

    # Training control
    total_steps: int = 1000
    validation_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    compile_model: bool = False

    # Early stopping
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


# -------------------------------------------------
# Dataset (np.memmap)
# -------------------------------------------------

class MemmapDataset:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.int32, mode="r")

    def sample_batch(self, batch_size: int, context_length: int, device: str):
        idx = np.random.randint(
            0, len(self.data) - context_length - 1, size=batch_size
        )

        x = np.stack([self.data[i:i+context_length] for i in idx])
        y = np.stack([self.data[i+1:i+context_length+1] for i in idx])

        return (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


# -------------------------------------------------
# Model 
# -------------------------------------------------

def build_model(config: TrainConfig):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.num_heads,
        dim_feedforward=4 * config.d_model,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)


# -------------------------------------------------
# Checkpointing
# -------------------------------------------------

def save_checkpoint(model, optimizer, step, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_step_{step}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )

    print(f"[Checkpoint saved] {path}")


# -------------------------------------------------
# Validation
# -------------------------------------------------

@torch.no_grad()
def evaluate(model, embedding, dataset, config, device):
    model.eval()

    x, y = dataset.sample_batch(
        config.batch_size,
        config.context_length,
        device,
    )

    x_emb = embedding(x)
    logits = model(x_emb)

    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
    )

    model.train()
    return loss.item()


# -------------------------------------------------
# Training Loop
# -------------------------------------------------

def train(config: TrainConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = MemmapDataset(config.data_path)
    val_data = (
        MemmapDataset(config.validation_data_path)
        if config.validation_data_path
        else None
    )

    model = build_model(config).to(device)

    # Add embedding layer for token IDs
    embedding = nn.Embedding(config.vocab_size, config.d_model).to(device)

    if config.compile_model:
        model = torch.compile(model)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(embedding.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for step in range(1, config.total_steps + 1):

        x, y = train_data.sample_batch(
            config.batch_size,
            config.context_length,
            device,
        )

        optimizer.zero_grad()

        x_emb = embedding(x)
        logits = model(x_emb)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        loss.backward()
        optimizer.step()

        # Console logging
        if step % 10 == 0:
            print(f"Step {step} | Train Loss: {loss.item():.4f}")


        # Validation
        if val_data and step % config.validation_interval == 0:

            val_loss = evaluate(model, embedding, val_data, config, device)

            print(
                f"Step {step} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if config.early_stopping_patience > 0:
                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        # Checkpointing
        if step % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, config)

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f} seconds.")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--validation-data-path", default=None)

    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")

    parser.add_argument("--compile-model", action="store_true")

    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)