import torch
import torch.nn as nn
from torchvision import models

class GDConv(nn.Module):
    """Global Depthwise Convolution - MobileFaceNet style head."""
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            groups=in_channels,  # Depthwise
            bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x.view(x.size(0), -1)  # Flatten

class MobileNetV3GDConv(nn.Module):
    """MobileNetV3-Small backbone with GDConv head."""
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        # Load MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Extract features (remove classifier)
        self.features = mobilenet.features  # Output: [B, 576, 7, 7]
        
        # GDConv head (MobileFaceNet-style)
        self.gdconv = GDConv(576, kernel_size=7)  # 576 channels, 7x7 spatial
        
        # Linear projection to embedding
        self.fc = nn.Linear(576, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x):
        x = self.features(x)       # [B, 576, 7, 7]
        x = self.gdconv(x)         # [B, 576]
        x = self.fc(x)             # [B, embedding_dim]
        x = self.bn(x)             # [B, embedding_dim]
        return x

def get_backbone(embedding_dim=512, pretrained=True):
    return MobileNetV3GDConv(embedding_dim=embedding_dim, pretrained=pretrained)

if __name__ == '__main__':
    model = get_backbone(512)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f'Input: {x.shape}')
    print(f'Output: {out.shape}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
