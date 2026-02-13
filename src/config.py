"""Configuration for pet face embedding training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data
    data_dir: Path = Path("/root/data")
    output_dir: Path = Path("outputs_224")

    # Model
    backbone: str = "mobilenetv3_small_100"
    embedding_dim: int = 128
    head_type: str = "gdconv"
    dropout: float = 0.4

    # ArcFace
    arcface_scale: float = 32.0
    arcface_margin: float = 0.4

    # Training
    batch_size: int = 96  # Slightly smaller due to larger images
    num_workers: int = 8
    epochs: int = 40
    lr: float = 5e-4
    weight_decay: float = 5e-4

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Input - INCREASED
    image_size: int = 224

    # Misc
    seed: int = 42
    device: str = "auto"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
