"""
Training script aligned with the original M5 configuration.

Highlights:
- Uses the original M5 model definition.
- Uses the original dataset pipeline and augmentations.

Usage:
  python train_modern.py --data data/meow_categories --epochs 30 --lr 3e-4
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.m5 import M5
from preprocessing.pipeline import load_dataset


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    print_every: int,
    save_every: int,
    checkpoint_dir: Path,
    early_stop_patience: int,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.NLLLoss()

    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        samples = 0
        train_bar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}",
            bar_format=bar_format,
        )
        for i, (wav, label) in enumerate(train_bar):
            wav, label = wav.to(device), label.to(device)
            optimizer.zero_grad()
            predictions = model(wav)
            class_predictions = torch.squeeze(torch.argmax(predictions, dim=-1))
            loss = criterion(predictions.squeeze(), label)
            loss.backward()
            optimizer.step()

            running_correct += (class_predictions == label).float().sum()
            running_loss += loss
            samples += label.shape[0]

            if i % print_every == 0:
                train_bar.set_postfix(
                    dict(
                        loss=f"{running_loss/samples:>7f}",
                        accuracy=f"{running_correct/samples:>7f}",
                    )
                )

        with torch.no_grad():
            running_loss = 0.0
            running_correct = 0.0
            samples = 0
            val_bar = tqdm(
                val_loader,
                desc=f"Test  Epoch {epoch}",
                bar_format=bar_format,
            )
            for i, (wav, label) in enumerate(val_bar):
                wav, label = wav.to(device), label.to(device)
                predictions = model(wav)
                class_predictions = torch.squeeze(torch.argmax(predictions, dim=-1))
                loss = criterion(predictions.squeeze(), label)
                running_correct += (class_predictions == label).float().sum()
                running_loss += loss
                samples += label.shape[0]

                if i % print_every == 0:
                    val_bar.set_postfix(
                        dict(
                            loss=f"{running_loss/samples:>7f}",
                            accuracy=f"{running_correct/samples:>7f}",
                        )
                    )

        val_acc = float(running_correct / samples) if samples > 0 else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            tqdm.write(
                f"Early stopping at epoch {epoch} (best val acc {best_val_acc:.4f})."
            )
            break

        if (epoch + 1) % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = checkpoint_dir / f"m5_epoch_{epoch+1}_{timestamp}.pt"
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            tqdm.write(f"Saved checkpoint to {ckpt_path}")

        scheduler.step()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M5-style model for cat emotions.")
    parser.add_argument("--data", required=True, help="Root folder with class subfolders of audio.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[80, 3, 3, 3])
    parser.add_argument("--filters", nargs="+", type=int, default=[32, 32, 64, 128])
    parser.add_argument("--strides", nargs="+", type=int, default=[16, 1, 1, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/Users/lhy/Documents/Github_Projects/checkpoints/cat_emtion"),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop after N epochs without validation accuracy improvement. 0 disables.",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    train_loader, val_loader = load_dataset(args.data, args.batch_size)
    num_classes = 0
    data_root = Path(args.data)
    for maybe_class in data_root.iterdir():
        if maybe_class.is_dir():
            num_classes += 1
    if num_classes == 0:
        raise RuntimeError(f"No class folders found under {data_root}")
    if not (len(args.kernel_sizes) == len(args.filters) == len(args.strides)):
        raise ValueError("kernel_sizes, filters, and strides must have the same length.")
    model = M5(
        kernel_sizes=args.kernel_sizes,
        filters=args.filters,
        strides=args.strides,
        num_classes=num_classes,
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        print_every=args.print_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()
