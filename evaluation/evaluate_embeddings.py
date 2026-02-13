"""Comprehensive embedding evaluation script.

Evaluates pet face embedding models on retrieval, clustering, ranking,
robustness, and visualization benchmarks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import io
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    roc_auc_score,
    roc_curve,
)
from scipy.stats import spearmanr

from src.config import Config
from src.model import PetFaceModel, PetFaceEmbedder
from src.dataset import load_dataset, get_val_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageDataset(Dataset):
    """Minimal dataset that returns image tensor + index."""

    def __init__(self, image_paths: list[Path], transform: A.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img, idx


@torch.no_grad()
def compute_embeddings(
    model: PetFaceEmbedder,
    image_paths: list[Path],
    transform: A.Compose,
    device: torch.device,
    batch_size: int = 128,
    tta: bool = False,
) -> np.ndarray:
    """Embed all images and return (N, D) numpy array.

    If tta=True, averages embeddings of original and horizontally flipped
    images, then re-normalizes to unit length.
    """
    model.eval()
    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_embs = []
    for imgs, _ in tqdm(loader, desc="Embedding"):
        imgs = imgs.to(device)
        embs = model(imgs)
        if tta:
            flipped = torch.flip(imgs, dims=[3])
            embs_flip = model(flipped)
            embs = (embs + embs_flip) / 2.0
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# A. Retrieval benchmarks
# ---------------------------------------------------------------------------

def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray,
                      ks: list[int] = [1, 5, 10]) -> dict:
    """Compute recall@k, MRR, NDCG@10 on the embedding matrix."""
    N = len(embeddings)
    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T
    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, -1.0)

    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    max_k = max(ks)

    for i in range(N):
        query_label = labels[i]
        ranked = np.argsort(-sim_matrix[i])  # descending
        ranked_labels = labels[ranked]

        # Recall@k
        for k in ks:
            if np.any(ranked_labels[:k] == query_label):
                recall[k] += 1.0

        # MRR
        correct_mask = ranked_labels == query_label
        if correct_mask.any():
            first_hit = np.argmax(correct_mask) + 1  # 1-indexed rank
            mrr_sum += 1.0 / first_hit

        # NDCG@10
        relevance = (ranked_labels[:10] == query_label).astype(float)
        dcg = np.sum(relevance / np.log2(np.arange(2, 12)))
        # Ideal DCG: all correct items first
        n_correct = int(correct_mask.sum())
        ideal_rel = np.zeros(10)
        ideal_rel[:min(n_correct, 10)] = 1.0
        idcg = np.sum(ideal_rel / np.log2(np.arange(2, 12)))
        if idcg > 0:
            ndcg_sum += dcg / idcg

    results = {}
    for k in ks:
        results[f"recall@{k}"] = recall[k] / N
    results["MRR"] = mrr_sum / N
    results["NDCG@10"] = ndcg_sum / N
    return results


# ---------------------------------------------------------------------------
# B. Clustering quality
# ---------------------------------------------------------------------------

def clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """KMeans clustering → NMI, ARI, purity."""
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    pred_clusters = kmeans.fit_predict(embeddings)

    nmi = normalized_mutual_info_score(labels, pred_clusters)
    ari = adjusted_rand_score(labels, pred_clusters)

    # Purity
    total = 0
    for c in range(n_clusters):
        mask = pred_clusters == c
        if mask.sum() == 0:
            continue
        cluster_labels = labels[mask]
        most_common = np.bincount(cluster_labels).max()
        total += most_common
    purity = total / len(labels)

    return {"NMI": nmi, "ARI": ari, "purity": purity}


# ---------------------------------------------------------------------------
# C. Similarity ranking (ROC-AUC, Spearman, EER)
# ---------------------------------------------------------------------------

def _sample_balanced_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_pairs: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample balanced positive/negative pairs via stratified sampling.

    Returns:
        sims: cosine similarities for each pair
        binary_labels: 1 for same identity, 0 for different
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    label_to_idx = {l: np.where(labels == l)[0] for l in unique_labels}
    valid_labels = [l for l in unique_labels if len(label_to_idx[l]) >= 2]

    half = num_pairs // 2
    idx1_list, idx2_list, pair_labels = [], [], []

    # Positive pairs (same identity)
    for _ in range(half):
        label = rng.choice(valid_labels)
        i, j = rng.choice(label_to_idx[label], size=2, replace=False)
        idx1_list.append(i)
        idx2_list.append(j)
        pair_labels.append(1)

    # Negative pairs (different identities)
    all_labels = list(unique_labels)
    for _ in range(half):
        l1, l2 = rng.choice(len(all_labels), size=2, replace=False)
        i = rng.choice(label_to_idx[all_labels[l1]])
        j = rng.choice(label_to_idx[all_labels[l2]])
        idx1_list.append(i)
        idx2_list.append(j)
        pair_labels.append(0)

    idx1 = np.array(idx1_list)
    idx2 = np.array(idx2_list)
    binary_labels = np.array(pair_labels)

    e1 = embeddings[idx1]
    e2 = embeddings[idx2]
    sims = np.sum(e1 * e2, axis=1) / (
        np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-8
    )

    return sims, binary_labels


def ranking_metrics(embeddings: np.ndarray, labels: np.ndarray,
                    num_pairs: int = 10000) -> dict:
    """Stratified pair sampling → ROC-AUC, Spearman, EER."""
    sims, binary_labels = _sample_balanced_pairs(embeddings, labels, num_pairs)

    # Skip if degenerate
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return {"ROC-AUC": float("nan"), "Spearman": float("nan"), "EER": float("nan")}

    auc = roc_auc_score(binary_labels, sims)
    spearman, _ = spearmanr(sims, binary_labels)

    # EER
    fpr, tpr, thresholds = roc_curve(binary_labels, sims)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return {
        "ROC-AUC": float(auc),
        "Spearman": float(spearman),
        "EER": float(eer),
        "positive_pairs": int(binary_labels.sum()),
        "negative_pairs": int((1 - binary_labels).sum()),
    }


# ---------------------------------------------------------------------------
# D. Triplet metrics
# ---------------------------------------------------------------------------

def triplet_metrics(embeddings: np.ndarray, labels: np.ndarray,
                    num_triplets: int = 20000) -> dict:
    """Sample triplets → violation rate, average margin."""
    rng = np.random.RandomState(42)
    unique_labels = np.unique(labels)
    label_to_idx = {l: np.where(labels == l)[0] for l in unique_labels}
    # Only labels with >= 2 samples
    valid_labels = [l for l in unique_labels if len(label_to_idx[l]) >= 2]

    violations = 0
    margins = []
    count = 0

    for _ in range(num_triplets):
        # Pick anchor label
        anchor_label = rng.choice(valid_labels)
        anchor_idx, pos_idx = rng.choice(label_to_idx[anchor_label], size=2, replace=False)

        # Pick negative label
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = rng.choice(unique_labels)
        neg_idx = rng.choice(label_to_idx[neg_label])

        a = embeddings[anchor_idx]
        p = embeddings[pos_idx]
        n = embeddings[neg_idx]

        sim_ap = np.dot(a, p) / (np.linalg.norm(a) * np.linalg.norm(p) + 1e-8)
        sim_an = np.dot(a, n) / (np.linalg.norm(a) * np.linalg.norm(n) + 1e-8)

        margin = sim_ap - sim_an
        margins.append(margin)
        if sim_ap < sim_an:
            violations += 1
        count += 1

    return {
        "triplet_violation_rate": violations / count,
        "average_margin": float(np.mean(margins)),
        "num_triplets": count,
    }


# ---------------------------------------------------------------------------
# E. Overfitting check
# ---------------------------------------------------------------------------

def _sample_by_identity(
    folders: list[Path],
    num_identities: int,
    min_images: int = 2,
    rng: random.Random | None = None,
) -> tuple[list[Path], list[int]]:
    """Sample identities (not images) and return all their images.

    Only includes identities with at least min_images so retrieval
    metrics are meaningful.
    """
    if rng is None:
        rng = random.Random(42)

    # Collect identities with enough images
    candidates = []
    for folder in folders:
        images = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if len(images) >= min_images:
            candidates.append((folder, images))

    rng.shuffle(candidates)
    selected = candidates[:num_identities]

    paths = []
    labels = []
    for label_idx, (folder, images) in enumerate(selected):
        for img in images:
            paths.append(img)
            labels.append(label_idx)

    return paths, labels


def overfitting_check(
    model: PetFaceEmbedder,
    train_paths: list[Path],
    train_folders: list[Path],
    test_paths: list[Path],
    test_folders: list[Path],
    transform: A.Compose,
    device: torch.device,
    num_identities: int = 500,
    tta: bool = False,
) -> dict:
    """Compare retrieval recall@1 on train vs test identities.

    Samples by identity (not by image) so each identity has multiple
    images in the sample, making R@1 meaningful.
    """
    rng = random.Random(42)

    train_sample_paths, train_sample_labels = _sample_by_identity(
        train_folders, num_identities, min_images=2, rng=rng)
    test_sample_paths, test_sample_labels = _sample_by_identity(
        test_folders, num_identities, min_images=2, rng=rng)

    train_embs = compute_embeddings(model, train_sample_paths, transform, device, tta=tta)
    train_labels = np.array(train_sample_labels)
    test_embs = compute_embeddings(model, test_sample_paths, transform, device, tta=tta)
    test_labels = np.array(test_sample_labels)

    train_r = retrieval_metrics(train_embs, train_labels, ks=[1])
    test_r = retrieval_metrics(test_embs, test_labels, ks=[1])

    return {
        "train_recall@1": train_r["recall@1"],
        "test_recall@1": test_r["recall@1"],
        "gap": train_r["recall@1"] - test_r["recall@1"],
        "train_identities": len(set(train_sample_labels)),
        "train_images": len(train_sample_paths),
        "test_identities": len(set(test_sample_labels)),
        "test_images": len(test_sample_paths),
    }


# ---------------------------------------------------------------------------
# F. Robustness probes
# ---------------------------------------------------------------------------

def apply_degradation(image: Image.Image, name: str) -> Image.Image:
    """Apply a single degradation to a PIL image."""
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3.5))
    elif name == "gaussian_noise":
        arr = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 0.05 * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif name == "jpeg_compression":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif name == "brightness_shift":
        arr = np.array(image).astype(np.float32)
        arr = np.clip(arr * 1.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif name == "partial_occlusion":
        img = image.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)
        # Black rectangle over 25% of image (center)
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        return img
    else:
        raise ValueError(f"Unknown degradation: {name}")


class DegradedDataset(Dataset):
    """Dataset that applies a degradation before the standard transform."""

    def __init__(self, image_paths: list[Path], degradation: str,
                 transform: A.Compose):
        self.image_paths = image_paths
        self.degradation = degradation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = apply_degradation(img, self.degradation)
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img, idx


@torch.no_grad()
def compute_degraded_embeddings(
    model: PetFaceEmbedder,
    image_paths: list[Path],
    degradation: str,
    transform: A.Compose,
    device: torch.device,
    batch_size: int = 128,
    tta: bool = False,
) -> np.ndarray:
    """Embed degraded images. If tta=True, averages original+flipped embeddings."""
    model.eval()
    dataset = DegradedDataset(image_paths, degradation, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_embs = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        embs = model(imgs)
        if tta:
            flipped = torch.flip(imgs, dims=[3])
            embs_flip = model(flipped)
            embs = (embs + embs_flip) / 2.0
            embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def robustness_probes(
    model: PetFaceEmbedder,
    image_paths: list[Path],
    labels: np.ndarray,
    clean_embeddings: np.ndarray,
    transform: A.Compose,
    device: torch.device,
    sample_size: int = 1000,
    tta: bool = False,
) -> dict:
    """Measure similarity drop and recall drop under degradations."""
    rng = random.Random(42)
    indices = list(range(len(image_paths)))
    rng.shuffle(indices)
    indices = indices[:sample_size]

    sample_paths = [image_paths[i] for i in indices]
    sample_labels = labels[indices]
    sample_clean = clean_embeddings[indices]

    degradations = [
        "gaussian_blur", "gaussian_noise", "jpeg_compression",
        "brightness_shift", "partial_occlusion",
    ]

    # Clean recall@1 on sample
    clean_retrieval = retrieval_metrics(sample_clean, sample_labels, ks=[1])
    clean_r1 = clean_retrieval["recall@1"]

    results = {}
    for deg in degradations:
        print(f"  Robustness probe: {deg}")
        deg_embs = compute_degraded_embeddings(
            model, sample_paths, deg, transform, device, tta=tta)

        # Cosine similarity between clean and degraded (same image)
        cos_sims = np.sum(sample_clean * deg_embs, axis=1) / (
            np.linalg.norm(sample_clean, axis=1) *
            np.linalg.norm(deg_embs, axis=1) + 1e-8
        )

        # Retrieval recall@1 on degraded embeddings
        deg_retrieval = retrieval_metrics(deg_embs, sample_labels, ks=[1])
        deg_r1 = deg_retrieval["recall@1"]

        results[deg] = {
            "mean_cosine_similarity": float(np.mean(cos_sims)),
            "similarity_drop": float(1.0 - np.mean(cos_sims)),
            "clean_recall@1": clean_r1,
            "degraded_recall@1": deg_r1,
            "recall_drop": clean_r1 - deg_r1,
        }

    return results


# ---------------------------------------------------------------------------
# G. Visualization: nearest-neighbor grid
# ---------------------------------------------------------------------------

def create_nn_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    image_paths: list[Path],
    output_path: Path,
    num_queries: int = 10,
    top_k: int = 5,
    thumb_size: int = 112,
    border: int = 4,
):
    """Create a grid of query + top-k nearest neighbors."""
    rng = random.Random(42)
    indices = list(range(len(embeddings)))
    rng.shuffle(indices)
    query_indices = indices[:num_queries]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, -1.0)

    cell = thumb_size + 2 * border
    cols = 1 + top_k
    rows = num_queries
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for row, qi in enumerate(query_indices):
        # Draw query image with blue border
        img = Image.open(image_paths[qi]).convert("RGB").resize((thumb_size, thumb_size))
        x = border
        y = row * cell + border
        draw.rectangle([x - border, y - border, x + thumb_size + border - 1,
                        y + thumb_size + border - 1], fill=(0, 120, 255))
        grid.paste(img, (x, y))

        # Top-k neighbors
        ranked = np.argsort(-sim_matrix[qi])[:top_k]
        for col_offset, ni in enumerate(ranked):
            correct = labels[ni] == labels[qi]
            color = (0, 200, 0) if correct else (220, 0, 0)
            nn_img = Image.open(image_paths[ni]).convert("RGB").resize(
                (thumb_size, thumb_size))
            x = (1 + col_offset) * cell + border
            y = row * cell + border
            draw.rectangle([x - border, y - border, x + thumb_size + border - 1,
                            y + thumb_size + border - 1], fill=color)
            grid.paste(nn_img, (x, y))

    grid.save(output_path)
    print(f"Saved nearest-neighbor visualization to {output_path}")


# ---------------------------------------------------------------------------
# H. Plots
# ---------------------------------------------------------------------------

def plot_similarity_distribution(
    embeddings: np.ndarray, labels: np.ndarray, output_path: Path,
    num_pairs: int = 10000,
):
    """Histogram of positive vs negative pair similarities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims, binary = _sample_balanced_pairs(embeddings, labels, num_pairs)
    binary = binary.astype(bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims[~binary], bins=100, alpha=0.6, label="Negative pairs", density=True)
    ax.hist(sims[binary], bins=100, alpha=0.6, label="Positive pairs", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Similarity Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved similarity distribution to {output_path}")


def plot_roc_curve(
    embeddings: np.ndarray, labels: np.ndarray, output_path: Path,
    num_pairs: int = 10000,
):
    """ROC curve for same/different classification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims, binary = _sample_balanced_pairs(embeddings, labels, num_pairs)

    if binary.sum() == 0 or binary.sum() == len(binary):
        print("Cannot plot ROC: all same or all different pairs")
        return

    fpr, tpr, _ = roc_curve(binary, sims)
    auc = roc_auc_score(binary, sims)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve to {output_path}")


def plot_robustness(robustness_results: dict, output_path: Path):
    """Bar chart of similarity drop per degradation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(robustness_results.keys())
    sim_drops = [robustness_results[n]["similarity_drop"] * 100 for n in names]
    recall_drops = [robustness_results[n]["recall_drop"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, sim_drops, width, label="Similarity Drop (%)")
    ax.bar(x + width / 2, recall_drops, width, label="Recall@1 Drop (%)")
    ax.set_ylabel("Drop (%)")
    ax.set_title("Robustness to Degradations")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved robustness chart to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate pet face embeddings")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to aligned image data (e.g. ~/cat_aligned)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for evaluation outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time flip averaging (average original + hflip embeddings)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    if args.tta:
        print("TTA enabled: will average original + horizontally flipped embeddings")

    # ---- Load dataset splits ----
    print("Loading dataset...")
    splits, identity_map, num_identities = load_dataset(
        args.data_dir, train_ratio=0.8, val_ratio=0.1, seed=42)

    train_paths, train_labels = splits["train"]
    test_paths, test_labels_raw, test_folders = splits["test"]

    # Build test labels from folder identity
    test_folder_to_label = {f.name: i for i, f in enumerate(test_folders)}
    test_labels = np.array([test_folder_to_label[p.parent.name] for p in test_paths])

    # Also get train folders for overfitting check
    # We need to recover train_folders from identity_map
    train_folder_names = sorted(identity_map.keys())
    train_folders = [args.data_dir / name for name in train_folder_names]

    print(f"Test: {len(test_paths)} images, {len(test_folders)} identities")

    # ---- Load model ----
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    print(f"Checkpoint epoch: {checkpoint['epoch']}, "
          f"val accuracy: {checkpoint['metrics'].get('accuracy', 'N/A')}")

    config = Config()
    config.image_size = args.image_size

    # Auto-detect backbone/head from saved config
    config_path = args.model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        config.backbone = saved.get("backbone", config.backbone)
        config.head_type = saved.get("head_type", config.head_type)
        config.image_size = saved.get("image_size", config.image_size)
        print(f"Loaded config: backbone={config.backbone}, head_type={config.head_type}, "
              f"image_size={config.image_size}")

    model = PetFaceModel(
        num_classes=num_identities,
        backbone=config.backbone,
        embedding_dim=config.embedding_dim,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        head_type=config.head_type,
        input_size=config.image_size,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    embedder = model.get_embedder()
    embedder.eval()

    transform = get_val_transforms(args.image_size)

    # ---- Compute test embeddings ----
    print("\nComputing test embeddings...")
    test_embs = compute_embeddings(embedder, test_paths, transform, device,
                                   batch_size=args.batch_size, tta=args.tta)
    print(f"Embedding shape: {test_embs.shape}")

    results = {}

    # ---- A. Retrieval benchmarks ----
    print("\n=== Retrieval Benchmarks ===")
    ret = retrieval_metrics(test_embs, test_labels)
    results["retrieval"] = ret
    for k, v in ret.items():
        print(f"  {k}: {v:.4f}")

    # ---- B. Clustering ----
    print("\n=== Clustering Metrics ===")
    clust = clustering_metrics(test_embs, test_labels)
    results["clustering"] = clust
    for k, v in clust.items():
        print(f"  {k}: {v:.4f}")

    # ---- C. Ranking ----
    print("\n=== Ranking Metrics ===")
    rank = ranking_metrics(test_embs, test_labels)
    results["ranking"] = rank
    for k, v in rank.items():
        print(f"  {k}: {v}")

    # ---- D. Triplet metrics ----
    print("\n=== Triplet Metrics ===")
    trip = triplet_metrics(test_embs, test_labels)
    results["triplet"] = trip
    for k, v in trip.items():
        print(f"  {k}: {v}")

    # ---- E. Overfitting check ----
    print("\n=== Overfitting Check ===")
    overfit = overfitting_check(
        embedder, train_paths, train_folders,
        test_paths, test_folders, transform, device, tta=args.tta)
    results["overfitting"] = overfit
    for k, v in overfit.items():
        print(f"  {k}: {v}")

    # ---- F. Robustness ----
    print("\n=== Robustness Probes ===")
    robust = robustness_probes(
        embedder, test_paths, test_labels, test_embs, transform, device,
        tta=args.tta)
    results["robustness"] = robust
    for deg, metrics in robust.items():
        print(f"  {deg}: sim_drop={metrics['similarity_drop']:.4f}, "
              f"recall_drop={metrics['recall_drop']:.4f}")

    # ---- Save results ----
    results["config"] = {"tta": args.tta}
    with open(args.output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {args.output_dir / 'eval_results.json'}")

    # ---- G. Visualization: NN grid ----
    print("\n=== Generating Visualizations ===")
    create_nn_visualization(
        test_embs, test_labels, test_paths,
        args.output_dir / "nearest_neighbors.png")

    # ---- H. Plots ----
    plot_similarity_distribution(
        test_embs, test_labels,
        args.output_dir / "similarity_distribution.png")

    plot_roc_curve(
        test_embs, test_labels,
        args.output_dir / "roc_curve.png")

    plot_robustness(robust, args.output_dir / "robustness.png")

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
