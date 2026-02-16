#!/usr/bin/env python3
"""
BYOL Self-Supervised Pretraining for Pet Face Recognition
MobileNetV3-Small + GDConv (MobileFaceNet-style head)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
import copy
import random
from PIL import Image

from src.mobilefacenet_backbone import get_backbone

# BYOL Components
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, feature_dim=512, hidden_dim=4096, proj_dim=256):
        super().__init__()
        self.online_encoder = backbone
        self.online_projector = MLP(feature_dim, hidden_dim, proj_dim)
        self.online_predictor = MLP(proj_dim, hidden_dim, proj_dim)
        
        self.target_encoder = copy.deepcopy(backbone)
        self.target_projector = MLP(feature_dim, hidden_dim, proj_dim)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target(self, tau=0.99):
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
    
    def forward(self, x1, x2):
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        p1 = self.online_predictor(self.online_projector(z1))
        p2 = self.online_predictor(self.online_projector(z2))
        
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))
        
        loss = self.loss_fn(p1, t2) + self.loss_fn(p2, t1)
        return loss / 2
    
    def loss_fn(self, p, t):
        p = F.normalize(p, dim=-1)
        t = F.normalize(t, dim=-1)
        return 2 - 2 * (p * t).sum(dim=-1).mean()

class PetDataset(Dataset):
    def __init__(self, root_dirs, transform1, transform2):
        self.images = []
        for root in root_dirs:
            root = Path(root)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                self.images.extend(root.rglob(ext))
        self.transform1 = transform1
        self.transform2 = transform2
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return self.transform1(img), self.transform2(img)

def get_transforms():
    base = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return base, base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--output', default='mobilenetv3_gdconv_byol.pth')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MobileNetV3-Small + GDConv head
    backbone = get_backbone(embedding_dim=args.embedding_dim, pretrained=True)
    print(f"Backbone: MobileNetV3-Small + GDConv, embedding_dim={args.embedding_dim}")
    print(f"Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    
    model = BYOL(backbone, feature_dim=args.embedding_dim).to(device)
    
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    t1, t2 = get_transforms()
    dataset = PetDataset(args.data, t1, t2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    print(f"\nStarting BYOL pretraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x1, x2 in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            loss = model(x1, x2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target(tau=0.99)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'backbone': model.online_encoder.state_dict(),
            }, f'byol_gdconv_checkpoint_epoch{epoch+1}.pth')
    
    torch.save(model.online_encoder.state_dict(), args.output)
    print(f"\nSaved pretrained backbone to {args.output}")

if __name__ == '__main__':
    main()
