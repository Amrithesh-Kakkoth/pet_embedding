"""Training script for pet face embedding model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.model import PetFaceModel
from src.dataset import (
    load_dataset,
    create_dataloaders,
    PairDataset,
    get_val_transforms,
)


def get_device(config: Config) -> torch.device:
    """Get the best available device."""
    if config.device != "auto":
        return torch.device(config.device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_margin_schedule(epoch: int, total_epochs: int, max_margin: float) -> float:
    """Gradually increase margin from 0 to max_margin over first 1/3 of training."""
    warmup_epochs = total_epochs // 3
    if epoch < warmup_epochs:
        return max_margin * (epoch / warmup_epochs)
    return max_margin


def get_lr_schedule(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Linear warmup for learning rate."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def train_epoch(
    model: PetFaceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        _, logits = model(images, labels)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def evaluate_verification(
    model: PetFaceModel,
    val_folders: list[Path],
    device: torch.device,
    num_pairs: int = 5000,
    image_size: int = 160,
) -> dict:
    """Evaluate using verification accuracy (same/different pairs)."""
    model.eval()

    # Create pair dataset
    pair_dataset = PairDataset(
        val_folders,
        num_pairs=num_pairs,
        transform=get_val_transforms(image_size),
    )

    pair_loader = DataLoader(
        pair_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    similarities = []
    labels = []

    for img1, img2, label in tqdm(pair_loader, desc="Evaluating"):
        img1 = img1.to(device)
        img2 = img2.to(device)

        emb1, _ = model(img1)
        emb2, _ = model(img2)

        # Cosine similarity
        sim = (emb1 * emb2).sum(dim=1).cpu().numpy()
        similarities.extend(sim)
        labels.extend(label.numpy())

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Debug: print similarity distribution
    pos_sims = similarities[labels == 1]
    neg_sims = similarities[labels == 0]
    print(f"  Similarity stats - Positive: mean={pos_sims.mean():.3f}, std={pos_sims.std():.3f}")
    print(f"  Similarity stats - Negative: mean={neg_sims.mean():.3f}, std={neg_sims.std():.3f}")

    # Find best threshold
    best_acc = 0
    best_threshold = 0

    for threshold in np.arange(-0.5, 1.0, 0.01):
        predictions = (similarities >= threshold).astype(int)
        acc = (predictions == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Calculate metrics at best threshold
    predictions = (similarities >= best_threshold).astype(int)

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": best_acc,
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_rate": recall,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "pos_sim_mean": float(pos_sims.mean()),
        "neg_sim_mean": float(neg_sims.mean()),
    }


def save_checkpoint(
    model: PetFaceModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def export_embedder(
    model: PetFaceModel,
    output_dir: Path,
    image_size: int = 160,
):
    """Export the embedder model for inference."""
    model.eval()
    embedder = model.get_embedder()
    embedder.eval()

    # Save PyTorch model
    torch.save(embedder.state_dict(), output_dir / "pet_embedder.pt")

    # Export to ONNX (on CPU to avoid device mismatch)
    embedder_cpu = embedder.cpu()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        embedder_cpu,
        dummy_input,
        output_dir / "pet_embedder.onnx",
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=18,
    )

    print(f"Exported models to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train pet face embedding model")
    parser.add_argument("--data-dir", type=Path, help="Path to dog dataset")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--backbone", type=str, help="Backbone model")
    parser.add_argument("--head-type", type=str, choices=["gdconv", "mlp"], help="Head type")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    parser.add_argument("--warm-restart", action="store_true",
                        help="Only load model weights (not optimizer), fresh LR schedule")
    args = parser.parse_args()

    # Load config
    config = Config()

    # Override with command line args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.backbone:
        config.backbone = args.backbone
    if args.head_type:
        config.head_type = args.head_type

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Setup device
    device = get_device(config)
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    splits, identity_map, num_identities = load_dataset(
        config.data_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    # Create dataloaders
    loaders = create_dataloaders(
        splits,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Create model
    print(f"Creating model with {num_identities} identities...")
    print(f"Head type: {config.head_type}")
    model = PetFaceModel(
        num_classes=num_identities,
        backbone=config.backbone,
        embedding_dim=config.embedding_dim,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        head_type=config.head_type,
        input_size=config.image_size,
        dropout=config.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedder_params = sum(p.numel() for p in model.embedder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Embedder parameters: {embedder_params:,}")

    # Optimizer with different learning rates for backbone vs head
    # MobileFaceNet trains from scratch → full LR for backbone
    # Pretrained backbones (timm) → 0.1x LR for backbone
    is_from_scratch = (config.backbone == "mobilefacenet")
    backbone_lr_factor = 1.0 if is_from_scratch else 0.1

    backbone_params = list(model.embedder.backbone.parameters())
    head_params = list(model.embedder.embedding.parameters()) + list(model.arcface.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config.lr * backbone_lr_factor},
        {"params": head_params, "lr": config.lr},
    ], weight_decay=config.weight_decay)

    # Cosine annealing with warmup
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - warmup_epochs,
        eta_min=1e-6,
    )

    # Label smoothing helps with generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.warm_restart:
            # Only load model weights, fresh optimizer + scheduler for warm restart
            print(f"Warm restart: keeping model weights, fresh optimizer (LR={config.lr})")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

    # Get validation folders for evaluation
    val_folders = splits["val"][2]

    # Training loop
    best_acc = 0
    history = []

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Using margin warmup over first {config.epochs // 3} epochs")
    print(f"Using LR warmup over first {warmup_epochs} epochs")

    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()

        # Update margin (warmup)
        current_margin = get_margin_schedule(epoch, config.epochs, config.arcface_margin)
        model.arcface.set_margin(current_margin)

        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = get_lr_schedule(epoch, warmup_epochs, config.lr)
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:  # Backbone params
                    param_group["lr"] = warmup_lr * backbone_lr_factor
                else:  # Head params
                    param_group["lr"] = warmup_lr

        # Train
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch + 1
        )

        # Update scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[1]["lr"]  # Head LR
        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {train_loss:.4f} - "
              f"Margin: {current_margin:.3f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")

        # Evaluate every 3 epochs (more frequent to track progress)
        metrics = {}
        if (epoch + 1) % 3 == 0 or epoch == config.epochs - 1:
            metrics = evaluate_verification(
                model, val_folders, device, num_pairs=5000, image_size=config.image_size
            )
            print(f"  Val Accuracy: {metrics['accuracy']:.4f} @ threshold {metrics['threshold']:.2f}")

            # Save best model
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    config.output_dir / "best_model.pt"
                )
                print(f"  New best model saved!")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": current_lr,
            "margin": current_margin,
            **metrics,
        })

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, metrics,
            config.output_dir / "last_checkpoint.pt"
        )

    # Save training history
    with open(config.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Export best model
    print("\nExporting best model...")
    checkpoint = torch.load(config.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    export_embedder(model, config.output_dir, config.image_size)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Models saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
