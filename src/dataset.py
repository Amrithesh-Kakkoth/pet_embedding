"""Dataset for pet face embedding training."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 160) -> A.Compose:
    """Augmentations for training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        # Geometric - pets tilt heads
        A.Rotate(limit=25, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            p=0.3,
        ),
        # Color - lighting variations
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5,
        ),
        A.ToGray(p=0.05),  # Helps generalize across fur colors
        # Quality - real-world phone cameras
        A.GaussianBlur(blur_limit=(3, 9), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
        # Low-res simulation
        A.Downscale(scale_range=(0.4, 0.8), p=0.15),
        # Lighting occlusion
        A.RandomShadow(p=0.1),
        # Occlusion
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            p=0.3,
        ),
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 160) -> A.Compose:
    """Transforms for validation/test (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class PetFaceDataset(Dataset):
    """Dataset for pet face images organized by identity folders."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: A.Compose | None = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        return image, label


def load_dataset(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[dict, dict, int]:
    """
    Load and split dataset by identity.

    Returns:
        splits: dict with 'train', 'val', 'test' containing (paths, labels)
        identity_map: dict mapping folder name to label index
        num_identities: total number of unique identities
    """
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)

    # Get all identity folders
    identity_folders = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(identity_folders)} identities")

    # Shuffle and split by identity
    random.shuffle(identity_folders)

    n_train = int(len(identity_folders) * train_ratio)
    n_val = int(len(identity_folders) * val_ratio)

    train_folders = identity_folders[:n_train]
    val_folders = identity_folders[n_train:n_train + n_val]
    test_folders = identity_folders[n_train + n_val:]

    print(f"Split: train={len(train_folders)}, val={len(val_folders)}, test={len(test_folders)}")

    # Create identity mapping (only for train identities for ArcFace)
    identity_map = {folder.name: idx for idx, folder in enumerate(train_folders)}

    def get_paths_and_labels(folders: list[Path], id_map: dict | None = None):
        paths = []
        labels = []

        for folder in folders:
            images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))

            for img_path in images:
                paths.append(img_path)
                if id_map is not None:
                    labels.append(id_map[folder.name])
                else:
                    labels.append(-1)  # Unknown label for val/test

        return paths, labels

    # Get paths and labels for each split
    train_paths, train_labels = get_paths_and_labels(train_folders, identity_map)
    val_paths, val_labels = get_paths_and_labels(val_folders)
    test_paths, test_labels = get_paths_and_labels(test_folders)

    print(f"Images: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels, val_folders),
        "test": (test_paths, test_labels, test_folders),
    }

    return splits, identity_map, len(train_folders)


def create_dataloaders(
    splits: dict,
    image_size: int = 160,
    batch_size: int = 128,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits."""

    train_paths, train_labels = splits["train"]

    train_dataset = PetFaceDataset(
        train_paths,
        train_labels,
        transform=get_train_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val/test loaders (for evaluation, we'll create them on-demand in evaluate)
    loaders = {
        "train": train_loader,
    }

    return loaders


class PairDataset(Dataset):
    """Dataset that returns pairs for verification evaluation."""

    def __init__(
        self,
        identity_folders: list[Path],
        num_pairs: int = 10000,
        transform: A.Compose | None = None,
    ):
        self.identity_folders = identity_folders
        self.num_pairs = num_pairs
        self.transform = transform

        # Build image lists per identity
        self.identity_images = {}
        for folder in identity_folders:
            images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            if len(images) >= 2:
                self.identity_images[folder.name] = images

        self.identity_list = list(self.identity_images.keys())
        self._generate_pairs()

    def _generate_pairs(self):
        """Generate positive and negative pairs."""
        self.pairs = []

        # Positive pairs (same identity)
        for _ in range(self.num_pairs // 2):
            identity = random.choice(self.identity_list)
            images = self.identity_images[identity]
            if len(images) >= 2:
                img1, img2 = random.sample(images, 2)
                self.pairs.append((img1, img2, 1))  # 1 = same

        # Negative pairs (different identity)
        for _ in range(self.num_pairs // 2):
            id1, id2 = random.sample(self.identity_list, 2)
            img1 = random.choice(self.identity_images[id1])
            img2 = random.choice(self.identity_images[id2])
            self.pairs.append((img1, img2, 0))  # 0 = different

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img1_path, img2_path, label = self.pairs[idx]

        img1 = np.array(Image.open(img1_path).convert("RGB"))
        img2 = np.array(Image.open(img2_path).convert("RGB"))

        if self.transform:
            img1 = self.transform(image=img1)["image"]
            img2 = self.transform(image=img2)["image"]

        return img1, img2, label
