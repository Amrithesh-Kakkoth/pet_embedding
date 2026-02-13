"""
MobileFaceNet Architecture for Pet Face Recognition

Based on the paper: "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices"

Key features:
- Depthwise separable convolutions
- Global Depthwise Convolution (GDConv) bottleneck
- Designed for face recognition
- ~1M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """Conv + BN + PReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    """Conv + BN (no activation)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = ConvBlock(in_channels, in_channels, 3, stride, 1, groups=in_channels)
        self.pointwise = LinearBlock(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Bottleneck(nn.Module):
    """MobileFaceNet Bottleneck Block (Inverted Residual)"""
    def __init__(self, in_channels, out_channels, stride, expansion):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion
        
        self.block = nn.Sequential(
            # Expansion
            ConvBlock(in_channels, hidden_dim, 1) if expansion != 1 else nn.Identity(),
            # Depthwise
            ConvBlock(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            # Projection
            LinearBlock(hidden_dim, out_channels, 1),
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class GDConv(nn.Module):
    """Global Depthwise Convolution"""
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        return self.bn(self.dwconv(x))


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet Architecture
    
    Input: 112x112 (original) or 224x224 (scaled)
    Output: 128/256/512-dim embedding
    """
    
    def __init__(self, embedding_dim: int = 128, input_size: int = 112, dropout: float = 0.4):
        super().__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        
        # Calculate GDConv kernel size based on input
        # After all convolutions, feature map is input_size / 16
        # For 112: 7x7, for 224: 14x14
        self.gdconv_size = input_size // 16
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)  # /2
        self.conv2 = DepthwiseSeparable(64, 64, 1)
        
        # Bottleneck blocks
        # (expansion, out_channels, num_blocks, stride)
        self.bottleneck_settings = [
            (2, 64, 5, 2),    # /4
            (4, 128, 1, 2),   # /8
            (2, 128, 6, 1),
            (4, 128, 1, 2),   # /16
            (2, 128, 2, 1),
        ]
        
        self.bottlenecks = self._make_bottlenecks()
        
        # Head
        self.conv3 = ConvBlock(128, 512, 1)
        self.gdconv = GDConv(512, self.gdconv_size)
        self.conv4 = LinearBlock(512, embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_bottlenecks(self):
        layers = []
        in_channels = 64
        
        for expansion, out_channels, num_blocks, stride in self.bottleneck_settings:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(Bottleneck(in_channels, out_channels, s, expansion))
                in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Bottlenecks
        x = self.bottlenecks(x)
        
        # Head
        x = self.conv3(x)
        x = self.gdconv(x)
        x = self.conv4(x)
        x = self.dropout(x)
        
        # Flatten and normalize
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class MobileFaceNetWithArcFace(nn.Module):
    """MobileFaceNet with ArcFace head for training."""
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        input_size: int = 112,
        dropout: float = 0.4,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
    ):
        super().__init__()
        
        self.backbone = MobileFaceNet(
            embedding_dim=embedding_dim,
            input_size=input_size,
            dropout=dropout,
        )
        
        self.arcface_scale = arcface_scale
        self.arcface_margin = arcface_margin
        
        # ArcFace classifier
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        
        if labels is None:
            return embeddings
        
        # ArcFace
        cos_theta = F.linear(embeddings, F.normalize(self.classifier.weight))
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        
        theta = torch.acos(cos_theta)
        target_theta = theta + self.arcface_margin
        
        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()
        cos_theta_m = torch.cos(target_theta)
        
        logits = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
        logits = logits * self.arcface_scale
        
        return embeddings, logits
    
    def get_embedder(self):
        """Return just the backbone for inference."""
        return self.backbone


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("MobileFaceNet Architecture Test")
    print("=" * 50)
    
    # Test 112x112
    model_112 = MobileFaceNet(embedding_dim=128, input_size=112)
    x = torch.randn(1, 3, 112, 112)
    model_112.eval()
    out = model_112(x)
    print(f"\nInput: 112x112")
    print(f"Output: {out.shape}")
    print(f"Parameters: {count_parameters(model_112):,}")
    
    # Test 224x224
    model_224 = MobileFaceNet(embedding_dim=128, input_size=224)
    x = torch.randn(1, 3, 224, 224)
    out = model_224(x)
    print(f"\nInput: 224x224")
    print(f"Output: {out.shape}")
    print(f"Parameters: {count_parameters(model_224):,}")
    
    # Test with ArcFace
    print("\nWith ArcFace head (57290 classes):")
    model_arc = MobileFaceNetWithArcFace(num_classes=57290, input_size=112)
    print(f"Total parameters: {count_parameters(model_arc):,}")
    print(f"Backbone parameters: {count_parameters(model_arc.backbone):,}")
