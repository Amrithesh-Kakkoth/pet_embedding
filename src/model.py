"""Pet face embedding model with ArcFace head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GDConv(nn.Module):
    """Global Depthwise Convolution from MobileFaceNet.

    Instead of global average pooling, uses depthwise conv with kernel size
    equal to the spatial dimensions. This learns position-aware weights,
    preserving more spatial information for recognition.
    """

    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # Depthwise
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, 1, 1)
        x = self.dwconv(x)
        x = self.bn(x)
        return x


class MobileFaceNetHead(nn.Module):
    """MobileFaceNet-style embedding head with GDConv.

    Architecture:
        GDConv (spatial pooling) -> 1x1 Conv (channel reduction) -> BN -> Flatten
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        feature_map_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Global Depthwise Convolution (replaces GAP)
        self.gdconv = GDConv(in_channels, kernel_size=feature_map_size)

        # Optional dropout before linear projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Linear projection via 1x1 conv (more efficient than flatten + linear)
        self.conv1x1 = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.gdconv(x)        # (B, C, 1, 1)
        x = self.dropout(x)
        x = self.conv1x1(x)       # (B, embedding_dim, 1, 1)
        x = self.bn(x)
        x = x.flatten(1)          # (B, embedding_dim)
        return x


class ArcFaceHead(nn.Module):
    """ArcFace angular margin loss for metric learning."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.base_margin = margin
        self.margin = 0.0  # Start with 0 margin, warm up
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def set_margin(self, margin: float):
        """Update margin (for warmup scheduling)."""
        self.margin = min(margin, self.base_margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weights)

        # Get angle
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Add margin to target class
        one_hot = F.one_hot(labels, num_classes=self.num_classes).bool()
        target_logits = torch.cos(theta + self.margin)

        # Combine
        logits = torch.where(one_hot, target_logits, cosine)

        return logits * self.scale


class PetFaceEmbedder(nn.Module):
    """Lightweight pet face embedding model.

    Supports two head architectures:
    - 'mlp': Standard MLP head after global average pooling (original)
    - 'gdconv': MobileFaceNet-style GDConv head (hybrid architecture)

    The GDConv head preserves spatial information better than GAP,
    which is beneficial for recognition tasks.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3_small_100",
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.4,
        head_type: str = "gdconv",  # 'mlp' or 'gdconv'
        input_size: int = 160,
    ):
        super().__init__()

        self.head_type = head_type
        self.embedding_dim = embedding_dim

        # MobileFaceNet is a complete embedder (has its own head + L2 norm)
        if backbone == "mobilefacenet":
            from src.mobilefacenet import MobileFaceNet
            self._is_mobilefacenet = True
            self.backbone = MobileFaceNet(
                embedding_dim=embedding_dim,
                input_size=input_size,
                dropout=dropout,
            )
            self.embedding = nn.Identity()
            return

        self._is_mobilefacenet = False

        if head_type == "gdconv":
            # For GDConv, we need feature maps, not pooled features
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",  # No pooling - get feature maps
                drop_rate=0.2,
            )

            # Determine feature map dimensions
            with torch.no_grad():
                dummy = torch.randn(1, 3, input_size, input_size)
                feat = self.backbone(dummy)
                _, channels, h, w = feat.shape
                assert h == w, f"Expected square feature map, got {h}x{w}"
                feature_map_size = h

            # MobileFaceNet-style head
            self.embedding = MobileFaceNetHead(
                in_channels=channels,
                embedding_dim=embedding_dim,
                feature_map_size=feature_map_size,
                dropout=dropout,
            )

        else:  # mlp head (original)
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier, keep GAP
                drop_rate=0.2,
            )

            # Get backbone output features
            with torch.no_grad():
                dummy = torch.randn(1, 3, input_size, input_size)
                backbone_dim = self.backbone(dummy).shape[1]

            # MLP embedding head
            self.embedding = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.PReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(512, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, '_is_mobilefacenet', False):
            # MobileFaceNet already L2-normalizes internally
            return self.backbone(x)
        features = self.backbone(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings


class PetFaceModel(nn.Module):
    """Full model with embedder + ArcFace head for training."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenetv3_small_100",
        embedding_dim: int = 128,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        pretrained: bool = True,
        head_type: str = "gdconv",
        input_size: int = 160,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.embedder = PetFaceEmbedder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            head_type=head_type,
            input_size=input_size,
            dropout=dropout,
        )

        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embeddings = self.embedder(x)

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return embeddings, logits

        return embeddings, None

    def get_embedder(self) -> PetFaceEmbedder:
        """Return just the embedder for inference."""
        return self.embedder
