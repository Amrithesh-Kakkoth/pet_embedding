"""Resume MobileFaceNet training from checkpoint"""

import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from mobilefacenet import MobileFaceNet
from dataset import load_dataset, PetFaceDataset, get_train_transforms, get_val_transforms


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, scale=32.0, margin=0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels=None):
        weight_norm = F.normalize(self.weight, dim=1)
        embeddings_norm = F.normalize(embeddings, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)
        
        if labels is None:
            return cosine * self.scale
            
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


def train_epoch(model, loader, optimizer, device, grad_clip=5.0):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), correct / total


def main():
    # Fixed settings matching original training
    data_dir = '/root/dog_data'
    output_dir = Path('outputs_mobilefacenet_v2')
    checkpoint_path = output_dir / 'last_checkpoint.pt'
    
    input_size = 224
    embedding_dim = 128
    batch_size = 128
    total_epochs = 50
    base_lr = 0.1
    weight_decay = 0.0005
    dropout = 0.4
    warmup_epochs = 3
    seed = 42
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
    
    # Load data
    print("\nLoading dataset...")
    splits, identity_map, num_identities = load_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, seed=seed)
    
    train_transform = get_train_transforms(image_size=input_size)
    val_transform = get_val_transforms(image_size=input_size)
    
    train_dataset = PetFaceDataset(splits['train'][0], splits['train'][1], transform=train_transform)
    val_dataset = PetFaceDataset(splits['val'][0], splits['val'][1], transform=val_transform)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # Create model
    backbone = MobileFaceNet(embedding_dim=embedding_dim, input_size=input_size, dropout=dropout)
    head = ArcFaceHead(in_features=embedding_dim, out_features=num_identities, scale=32.0, margin=0.4)
    model = MobileFaceNetWithHead(backbone, head)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # LR schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(start_epoch):
        scheduler.step()
    
    # Load history
    history_path = output_dir / 'history.json'
    with open(history_path) as f:
        history = json.load(f)
    
    best_acc = max((h.get('val_acc', 0) for h in history), default=0)
    
    print(f"\nResuming training from epoch {start_epoch} to {total_epochs}")
    
    for epoch in range(start_epoch, total_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{total_epochs} (lr={current_lr:.6f})")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        epoch_data = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'lr': current_lr}
        history.append(epoch_data)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'backbone_state_dict': backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / 'last_checkpoint.pt')
        
        # Save backbone
        torch.save(backbone.state_dict(), output_dir / 'pet_embedder_mobilefacenet.pt')
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")


if __name__ == '__main__':
    main()
