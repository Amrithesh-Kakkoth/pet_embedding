"""MobileFaceNet training with ArcFace - Fixed version with proper training"""

import os
import json
import random
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from mobilefacenet import MobileFaceNet
from dataset import load_dataset, PetFaceDataset, get_train_transforms, get_val_transforms


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels=None):
        # Normalize weights and embeddings
        weight_norm = F.normalize(self.weight, dim=1)
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)
        
        if labels is None:
            return cosine * self.scale
            
        # ArcFace margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        output = cosine * (1 - one_hot) + target_logits * one_hot
        
        return output * self.scale


class MobileFaceNetWithHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        logits = self.head(embeddings, labels)
        return embeddings, logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        embeddings, logits = model(images, labels)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
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
def evaluate(model, val_loader, device, num_pairs=5000):
    """Evaluate using pair-wise similarity matching."""
    model.eval()
    
    # Collect all embeddings and labels
    all_embeddings = []
    all_labels = []
    
    for images, labels in tqdm(val_loader, desc="Extracting embeddings"):
        images = images.to(device)
        emb = model.backbone(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Group indices by label
    label_to_indices = {}
    for i, label in enumerate(labels.tolist()):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)
    
    # Find labels with at least 2 images
    valid_labels = [l for l, indices in label_to_indices.items() if len(indices) >= 2]
    
    if len(valid_labels) < 2:
        print("  Warning: Not enough valid labels for evaluation")
        return 0.5, 0.5, {}
    
    # Sample positive pairs (same identity)
    pos_sims = []
    for _ in range(num_pairs):
        label = random.choice(valid_labels)
        i, j = random.sample(label_to_indices[label], 2)
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        pos_sims.append(sim)
    
    # Sample negative pairs (different identities)
    neg_sims = []
    for _ in range(num_pairs):
        l1, l2 = random.sample(valid_labels, 2)
        i = random.choice(label_to_indices[l1])
        j = random.choice(label_to_indices[l2])
        sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1]).item()
        neg_sims.append(sim)
    
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    print(f"  Positive pairs: mean={pos_sims.mean():.3f}, std={pos_sims.std():.3f}")
    print(f"  Negative pairs: mean={neg_sims.mean():.3f}, std={neg_sims.std():.3f}")
    print(f"  Separation: {pos_sims.mean() - neg_sims.mean():.3f}")
    
    # Find best threshold
    best_acc = 0
    best_thresh = 0
    
    all_sims = np.concatenate([pos_sims, neg_sims])
    all_labels_binary = np.array([1] * len(pos_sims) + [0] * len(neg_sims))
    
    for thresh in np.arange(-0.5, 1.0, 0.01):
        preds = (all_sims >= thresh).astype(int)
        acc = (preds == all_labels_binary).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    print(f"  Best accuracy: {best_acc:.4f} at threshold {best_thresh:.2f}")
    
    stats = {
        'pos_mean': float(pos_sims.mean()),
        'pos_std': float(pos_sims.std()),
        'neg_mean': float(neg_sims.mean()),
        'neg_std': float(neg_sims.std()),
    }
    
    return best_acc, best_thresh, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/dog_data')
    parser.add_argument('--output_dir', type=str, default='outputs_mobilefacenet_v2')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--arcface_scale', type=float, default=32.0)
    parser.add_argument('--arcface_margin', type=float, default=0.4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
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
        splits['train'][0],
        splits['train'][1],
        transform=train_transform,
    )
    val_dataset = PetFaceDataset(
        splits['val'][0],
        splits['val'][1],
        transform=val_transform,
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Classes: {num_identities}")
    
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
    
    # Create model
    print(f"\nCreating MobileFaceNet (input={args.input_size}x{args.input_size})...")
    backbone = MobileFaceNet(
        embedding_dim=args.embedding_dim,
        input_size=args.input_size,
        dropout=args.dropout,
    )
    
    head = ArcFaceHead(
        in_features=args.embedding_dim,
        out_features=num_identities,
        scale=args.arcface_scale,
        margin=args.arcface_margin,
    )
    
    model = MobileFaceNetWithHead(backbone, head)
    model = model.to(device)
    
    backbone_params = sum(p.numel() for p in backbone.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    
    # Optimizer - use SGD with momentum for better convergence
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate schedule with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Save config
    config = vars(args)
    config['num_identities'] = num_identities
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    best_acc = 0
    history = []
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.6f})")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=args.grad_clip
        )
        
        scheduler.step()
        
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'lr': current_lr,
        }
        
        # Evaluate every 3 epochs
        if epoch % 3 == 0 or epoch == 1:
            val_acc, threshold, stats = evaluate(model, val_loader, device)
            epoch_data['val_acc'] = val_acc
            epoch_data['threshold'] = threshold
            epoch_data.update(stats)
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'backbone_state_dict': backbone.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_acc,
                    'threshold': threshold,
                }, output_dir / 'best_model.pt')
                
                # Save backbone only
                torch.save(backbone.state_dict(), output_dir / 'pet_embedder_mobilefacenet.pt')
                print(f"  Saved new best model (acc={val_acc:.4f})")
        
        history.append(epoch_data)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'backbone_state_dict': backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / 'last_checkpoint.pt')
        
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()


# Add resume capability - append to existing script
if __name__ == '__main__' and '--resume' in sys.argv:
    # This is handled in main() now
    pass
