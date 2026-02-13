#!/usr/bin/env python3
"""
Training script for MobileFaceNet on Pet Face Dataset

Architecture: Pure MobileFaceNet (not MobileNetV3)
Loss: ArcFace
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.mobilefacenet import MobileFaceNetWithArcFace
from src.dataset import load_dataset, PetFaceDataset, get_train_transforms, get_val_transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        embeddings, logits = model(images, labels)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, val_loader, device, thresholds=None):
    model.eval()
    
    embeddings_list = []
    labels_list = []
    
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        emb = model.backbone(images)
        embeddings_list.append(emb.cpu())
        labels_list.append(labels)
    
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Sample pairs for evaluation
    n_samples = min(5000, len(embeddings))
    
    pos_sims = []
    neg_sims = []
    
    # Group by label
    label_to_indices = {}
    for i, label in enumerate(labels.tolist()):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    # Sample positive pairs
    for label, idx_list in label_to_indices.items():
        if len(idx_list) >= 2:
            for i in range(min(3, len(idx_list))):
                for j in range(i + 1, min(4, len(idx_list))):
                    sim = F.cosine_similarity(
                        embeddings[idx_list[i]].unsqueeze(0),
                        embeddings[idx_list[j]].unsqueeze(0)
                    ).item()
                    pos_sims.append(sim)
    
    # Sample negative pairs
    all_labels = list(label_to_indices.keys())
    if len(all_labels) >= 2:
        for _ in range(len(pos_sims)):
            l1, l2 = random.sample(all_labels, 2)
            i1 = random.choice(label_to_indices[l1])
            i2 = random.choice(label_to_indices[l2])
            sim = F.cosine_similarity(
                embeddings[i1].unsqueeze(0),
                embeddings[i2].unsqueeze(0)
            ).item()
            neg_sims.append(sim)
    else:
        print("  Warning: Not enough labels for negative pair sampling")
    
    pos_sims = np.array(pos_sims) if pos_sims else np.array([0.0])
    neg_sims = np.array(neg_sims) if neg_sims else np.array([0.0])
    
    if len(pos_sims) > 1:
        print(f"  Similarity stats - Positive: mean={pos_sims.mean():.3f}, std={pos_sims.std():.3f}")
    if len(neg_sims) > 1:
        print(f"  Similarity stats - Negative: mean={neg_sims.mean():.3f}, std={neg_sims.std():.3f}")
    
    # Find best threshold
    if thresholds is None:
        thresholds = np.arange(0.0, 0.5, 0.01)
    
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        tp = (pos_sims >= thresh).sum()
        tn = (neg_sims < thresh).sum()
        acc = (tp + tn) / (len(pos_sims) + len(neg_sims))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_acc, best_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/dog_data')
    parser.add_argument('--output_dir', type=str, default='outputs_mobilefacenet')
    parser.add_argument('--input_size', type=int, default=112)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--arcface_scale', type=float, default=64.0)
    parser.add_argument('--arcface_margin', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    splits, identity_map, num_identities = load_dataset(
        args.data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
    )
    
    # Create transforms
    train_transform = get_train_transforms(image_size=args.input_size)
    val_transform = get_val_transforms(image_size=args.input_size)
    
    train_dataset = PetFaceDataset(
        splits['train'][0],  # paths
        splits['train'][1],  # labels
        transform=train_transform,
    )
    val_dataset = PetFaceDataset(
        splits['val'][0],
        splits['val'][1],
        transform=val_transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Classes: {num_identities}")
    
    # Create model
    print(f"\nCreating MobileFaceNet (input={args.input_size}x{args.input_size})...")
    model = MobileFaceNetWithArcFace(
        num_classes=num_identities,
        embedding_dim=args.embedding_dim,
        input_size=args.input_size,
        dropout=args.dropout,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"Backbone parameters: {n_backbone:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Save config
    config = vars(args)
    config['num_classes'] = num_identities
    config['n_params'] = n_params
    config['n_backbone'] = n_backbone
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        scheduler.step()
        
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - TrainAcc: {train_acc:.4f} - LR: {lr:.6f} - Time: {elapsed:.1f}s")
        
        # Evaluate every 3 epochs
        if epoch % 3 == 0 or epoch == args.epochs:
            val_acc, threshold = evaluate(model, val_loader, device)
            print(f"  Val Accuracy: {val_acc:.4f} @ threshold {threshold:.2f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': {'accuracy': val_acc, 'threshold': threshold},
                }, output_dir / 'best_model.pt')
                print(f"  New best model saved!")
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'threshold': threshold,
                'lr': lr,
            })
        else:
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'lr': lr,
            })
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / 'last_checkpoint.pt')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Export embedder
    print("\nExporting embedder...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    embedder = model.get_embedder()
    torch.save(embedder.state_dict(), output_dir / 'pet_embedder_mobilefacenet.pt')
    
    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
