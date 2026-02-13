"""Quick test to verify the training setup works."""

import sys
from pathlib import Path

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    import torch
    import timm
    import albumentations
    from PIL import Image
    import numpy as np
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print("  Imports OK")
    return True

def test_model():
    """Test model creation."""
    print("\nTesting model...")
    from model import PetFaceEmbedder, PetFaceModel
    import torch

    # Test embedder
    embedder = PetFaceEmbedder(embedding_dim=128)
    dummy = torch.randn(2, 3, 160, 160)
    out = embedder(dummy)
    print(f"  Embedder output shape: {out.shape}")
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    # Test full model
    model = PetFaceModel(num_classes=100, embedding_dim=128)
    emb, logits = model(dummy, torch.tensor([0, 1]))
    print(f"  Full model embedding: {emb.shape}, logits: {logits.shape}")

    # Count parameters
    params = sum(p.numel() for p in embedder.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in embedder.parameters()) / 1024 / 1024
    print(f"  Embedder params: {params:,} ({size_mb:.2f} MB)")
    print("  Model OK")
    return True

def test_dataset(data_dir: Path, num_samples: int = 10):
    """Test dataset loading."""
    print(f"\nTesting dataset from {data_dir}...")
    from dataset import load_dataset, PetFaceDataset, get_train_transforms
    import torch

    # Load a small subset
    splits, identity_map, num_ids = load_dataset(
        data_dir,
        train_ratio=0.01,  # Just 1% for testing
        val_ratio=0.005,
        seed=42,
    )

    train_paths, train_labels = splits["train"]
    print(f"  Train samples: {len(train_paths)}")
    print(f"  Unique identities: {num_ids}")

    # Test dataset
    dataset = PetFaceDataset(
        train_paths[:num_samples],
        train_labels[:num_samples],
        transform=get_train_transforms(160),
    )

    img, label = dataset[0]
    print(f"  Sample image shape: {img.shape}")
    print(f"  Sample label: {label}")

    # Test dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch_img, batch_labels = next(iter(loader))
    print(f"  Batch shape: {batch_img.shape}")
    print("  Dataset OK")
    return True, num_ids

def test_training_step(data_dir: Path):
    """Test a single training step."""
    print("\nTesting training step...")
    import torch
    import torch.nn as nn
    from model import PetFaceModel
    from dataset import load_dataset, PetFaceDataset, get_train_transforms

    # Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Using device: {device}")

    # Load minimal data
    splits, _, num_ids = load_dataset(data_dir, train_ratio=0.005, val_ratio=0.001, seed=42)
    train_paths, train_labels = splits["train"]

    # Create dataset with just 32 samples
    dataset = PetFaceDataset(
        train_paths[:32],
        train_labels[:32],
        transform=get_train_transforms(160),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Create model
    model = PetFaceModel(num_classes=num_ids, embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Single training step
    model.train()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    embeddings, logits = model(images, labels)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print("  Training step OK")
    return True

def test_inference():
    """Test inference pipeline components."""
    print("\nTesting inference components...")
    import torch
    from model import PetFaceEmbedder
    import torch.nn.functional as F

    embedder = PetFaceEmbedder(embedding_dim=128)
    embedder.eval()

    # Simulate two face embeddings
    with torch.no_grad():
        face1 = torch.randn(1, 3, 160, 160)
        face2 = torch.randn(1, 3, 160, 160)

        emb1 = embedder(face1)
        emb2 = embedder(face2)

        similarity = F.cosine_similarity(emb1, emb2).item()

    print(f"  Embedding 1 norm: {emb1.norm().item():.4f} (should be ~1.0)")
    print(f"  Similarity between random faces: {similarity:.4f}")
    print("  Inference OK")
    return True

def main():
    data_dir = Path.home() / "Downloads" / "dog"

    print("=" * 50)
    print("Pet Face Embedding - Setup Test")
    print("=" * 50)

    try:
        test_imports()
        test_model()
        test_dataset(data_dir)
        test_training_step(data_dir)
        test_inference()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYou can now run the full training:")
        print("  python train.py --epochs 30")
        print("\nOr for a quick test run:")
        print("  python train.py --epochs 2 --batch-size 64")
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
