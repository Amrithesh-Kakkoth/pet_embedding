#!/usr/bin/env python3
"""Cluster Battersea pet images by embedding similarity.

Detects faces, computes embeddings, clusters by identity using HDBSCAN,
and saves images into per-cluster directories. Also compares predicted
clusters against ground truth folder structure.

Usage:
    python cluster_battersea.py --species cat
    python cluster_battersea.py --species dog
"""

import argparse
import json
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/root/pipeline")

from face_loader import load_face_model, DEFAULT_FACE_MODEL
from face_align import FaceAligner

import onnxruntime as ort
import hdbscan
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_embedder(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess(bgr_image):
    img = cv2.resize(bgr_image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis]


def embed(session, input_name, face_bgr):
    inp = preprocess(face_bgr)
    embedding = session.run(None, {input_name: inp})[0][0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def detect_faces_yolo(model, img, conf_thresh=0.3, device="cuda"):
    import torch
    h0, w0 = img.shape[:2]
    target_size = 640
    scale = target_size / max(h0, w0)
    if scale < 1:
        img_resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    else:
        img_resized = img
        scale = 1.0

    h, w = img_resized.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    img_padded = cv2.copyMakeBorder(
        img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    img_t = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)[0]

    from utils_face.general import non_max_suppression_face
    pred = non_max_suppression_face(pred, conf_thresh, 0.5)

    faces = []
    if pred[0] is not None and len(pred[0]) > 0:
        det = pred[0].cpu().numpy()
        for d in det:
            box = d[:4] / scale
            conf = d[4]
            landmarks = d[5:11].reshape(3, 2) / scale if len(d) >= 11 else None
            faces.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "landmarks": landmarks,
            })
    return faces


def align_and_crop(img, face, aligner, output_size=224):
    bbox = face["bbox"]
    landmarks = face.get("landmarks")

    if landmarks is not None and len(landmarks) >= 2:
        aligned, info = aligner.align(img, landmarks, [int(v) for v in bbox])
        if aligned is not None:
            return aligned

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * 0.2)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size))


def process_and_embed(data_dir, species, face_model, aligner,
                      embedder_session, input_name, device="cuda"):
    """Detect faces and compute embeddings for all images."""
    species_dir = Path(data_dir) / species
    identity_dirs = sorted(
        [d for d in species_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    print(f"Found {len(identity_dirs)} {species} identities")

    records = []
    for identity_dir in identity_dirs:
        identity = identity_dir.name
        images = sorted(
            [f for f in identity_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        )
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = detect_faces_yolo(face_model, img, conf_thresh=0.3, device=device)

            if len(faces) == 0:
                crop = cv2.resize(img, (224, 224))
            else:
                best_face = max(faces, key=lambda f: f["confidence"])
                crop = align_and_crop(img, best_face, aligner)
                if crop is None:
                    crop = cv2.resize(img, (224, 224))

            emb = embed(embedder_session, input_name, crop)
            records.append({
                "identity": identity,
                "image_path": str(img_path),
                "embedding": emb,
            })

    print(f"Embedded {len(records)} images")
    return records


def cluster_and_save(records, output_dir, min_cluster_size=2, min_samples=2):
    """Cluster embeddings with HDBSCAN and save images into cluster dirs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    embeddings = np.array([r["embedding"] for r in records])
    gt_identities = [r["identity"] for r in records]

    # Cosine distance matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    cosine_sim = normed @ normed.T
    distance_matrix = np.clip(1.0 - cosine_sim, 0.0, 2.0).astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    pred_labels = clusterer.fit_predict(distance_matrix)

    n_clusters = len(set(pred_labels) - {-1})
    n_noise = (pred_labels == -1).sum()
    print(f"\nHDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}): "
          f"{n_clusters} clusters, {n_noise} noise points")

    # Save images into cluster directories
    for i, record in enumerate(records):
        cluster_id = pred_labels[i]
        if cluster_id == -1:
            cluster_dir = output_dir / "unclustered"
        else:
            cluster_dir = output_dir / f"cluster_{cluster_id:03d}"

        cluster_dir.mkdir(exist_ok=True)

        src = Path(record["image_path"])
        # Prefix with ground truth identity for easy visual checking
        dst_name = f"{record['identity']}__{src.name}"
        dst = cluster_dir / dst_name
        shutil.copy2(src, dst)

    # Compute clustering quality vs ground truth
    # Only for non-noise points
    mask = pred_labels != -1
    if mask.sum() > 0:
        gt_labels_str = [gt_identities[i] for i in range(len(records)) if mask[i]]
        unique_gt = sorted(set(gt_labels_str))
        gt_map = {name: idx for idx, name in enumerate(unique_gt)}
        gt_numeric = [gt_map[g] for g in gt_labels_str]
        pred_numeric = pred_labels[mask]

        nmi = normalized_mutual_info_score(gt_numeric, pred_numeric)
        ari = adjusted_rand_score(gt_numeric, pred_numeric)

        # Purity
        cluster_ids = set(pred_numeric)
        total_correct = 0
        for c in cluster_ids:
            c_mask = pred_numeric == c
            c_gt = [gt_numeric[i] for i in range(len(gt_numeric)) if c_mask[i]]
            most_common = max(set(c_gt), key=c_gt.count)
            total_correct += c_gt.count(most_common)
        purity = total_correct / len(gt_numeric)
    else:
        nmi = ari = purity = 0.0

    # Per-cluster breakdown
    cluster_contents = defaultdict(lambda: defaultdict(int))
    for i, record in enumerate(records):
        c = pred_labels[i]
        label = "unclustered" if c == -1 else f"cluster_{c:03d}"
        cluster_contents[label][record["identity"]] += 1

    return {
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "n_total": len(records),
        "n_clustered": int(mask.sum()),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "NMI": float(nmi),
        "ARI": float(ari),
        "purity": float(purity),
        "cluster_contents": {k: dict(v) for k, v in sorted(cluster_contents.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=["cat", "dog"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/root/battersea-pets"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cat-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_cat_v2/pet_embedder.onnx"))
    parser.add_argument("--dog-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_dog_v2/pet_embedder.onnx"))
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="HDBSCAN min_samples")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(f"/root/pet_embedding/battersea_{args.species}_clusters")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading face detection model...")
    sys.path.insert(0, "/root/pipeline/yolov5_face")
    face_model = load_face_model(DEFAULT_FACE_MODEL, device=device, cleanup_aliases=False)
    face_model = face_model.to(device)
    aligner = FaceAligner()

    onnx_path = args.cat_model if args.species == "cat" else args.dog_model
    print(f"Loading embedder: {onnx_path}")
    embedder_session, input_name = load_embedder(onnx_path)

    # Process images
    print(f"\nProcessing {args.species} images...")
    records = process_and_embed(
        args.data_dir, args.species, face_model, aligner,
        embedder_session, input_name, device=device,
    )

    # Cluster and save
    print(f"\nClustering and saving to {args.output_dir}...")
    metrics = cluster_and_save(records, args.output_dir,
                               min_cluster_size=args.min_cluster_size,
                               min_samples=args.min_samples)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"BATTERSEA {args.species.upper()} — CLUSTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total images:    {metrics['n_total']}")
    print(f"  Clusters found:  {metrics['n_clusters']}")
    print(f"  Clustered:       {metrics['n_clustered']}")
    print(f"  Unclustered:     {metrics['n_noise']}")
    print(f"  NMI:             {metrics['NMI']:.3f}")
    print(f"  ARI:             {metrics['ARI']:.3f}")
    print(f"  Purity:          {metrics['purity']:.3f}")

    # Show cluster breakdown
    print(f"\n  Cluster breakdown:")
    for cluster_name, identities in sorted(metrics["cluster_contents"].items()):
        total = sum(identities.values())
        dominant = max(identities, key=identities.get)
        dominant_pct = identities[dominant] / total * 100
        id_str = ", ".join(f"{k}({v})" for k, v in sorted(identities.items(), key=lambda x: -x[1]))
        pure = "PURE" if len(identities) == 1 else "MIXED"
        print(f"    {cluster_name:<20} [{pure:5s}] {total:3d} imgs — {id_str}")

    # Save metrics
    with open(args.output_dir / "cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
