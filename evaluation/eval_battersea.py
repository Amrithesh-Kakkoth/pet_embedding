#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Evaluate pet embedding models on Battersea Pets dataset.

Uses the full pipeline (face detection + alignment + ONNX embedding) to test
on real-world shelter photos from Battersea. Ground truth identity comes from
the folder structure.

Usage:
    python eval_battersea.py --species cat
    python eval_battersea.py --species dog
"""

import argparse
import json
import cv2
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/root/pipeline")

from face_loader import load_face_model, DEFAULT_FACE_MODEL
from face_align import FaceAligner

import onnxruntime as ort

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_embedder(onnx_path):
    """Load ONNX embedding model."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess(bgr_image):
    """Resize, BGR->RGB, normalize, HWC->CHW, add batch dim."""
    img = cv2.resize(bgr_image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis]


def embed(session, input_name, face_bgr):
    """Get L2-normalized embedding."""
    inp = preprocess(face_bgr)
    embedding = session.run(None, {input_name: inp})[0][0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def detect_faces_yolo(model, img, conf_thresh=0.3, device="cuda"):
    """Detect faces using YOLOv5-face model."""
    import torch

    h0, w0 = img.shape[:2]
    target_size = 640
    scale = target_size / max(h0, w0)
    if scale < 1:
        img_resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    else:
        img_resized = img
        scale = 1.0

    # Pad to square
    h, w = img_resized.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    img_padded = cv2.copyMakeBorder(
        img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # To tensor
    img_t = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)[0]

    # NMS
    from utils_face.general import non_max_suppression_face

    pred = non_max_suppression_face(pred, conf_thresh, 0.5)

    faces = []
    if pred[0] is not None and len(pred[0]) > 0:
        det = pred[0].cpu().numpy()
        for d in det:
            box = d[:4] / scale
            conf = d[4]
            landmarks = d[5:11].reshape(3, 2) / scale if len(d) >= 11 else None
            faces.append(
                {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "landmarks": landmarks,
                }
            )

    return faces


def align_and_crop(img, face, aligner, output_size=224):
    """Align and crop a detected face."""
    bbox = face["bbox"]
    landmarks = face.get("landmarks")

    if landmarks is not None and len(landmarks) >= 2:
        aligned, info = aligner.align(img, landmarks, [int(v) for v in bbox])
        if aligned is not None:
            return aligned

    # Fallback: just crop the bbox with padding
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    # Add 20% padding
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


def process_dataset(data_dir, species, face_model, aligner, embedder_session,
                    input_name, device="cuda"):
    """Process all images, detect faces, compute embeddings.

    Returns:
        results: list of dicts with identity, image_path, embedding, face_count
    """
    species_dir = Path(data_dir) / species
    if not species_dir.exists():
        print(f"ERROR: {species_dir} does not exist")
        return []

    identity_dirs = sorted(
        [d for d in species_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    print(f"Found {len(identity_dirs)} {species} identities")

    results = []
    no_face_count = 0
    multi_face_count = 0
    total_images = 0

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        images = sorted(
            [f for f in identity_dir.iterdir()
             if f.suffix.lower() in IMAGE_EXTENSIONS]
        )

        for img_path in images:
            total_images += 1
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: Could not read {img_path}")
                continue

            faces = detect_faces_yolo(face_model, img, conf_thresh=0.3, device=device)

            if len(faces) == 0:
                # No face detected — use entire image as fallback
                no_face_count += 1
                crop = cv2.resize(img, (224, 224))
                emb = embed(embedder_session, input_name, crop)
                results.append({
                    "identity": identity,
                    "image_path": str(img_path),
                    "embedding": emb,
                    "face_detected": False,
                    "face_count": 0,
                })
                continue

            if len(faces) > 1:
                multi_face_count += 1

            # Use highest confidence face
            best_face = max(faces, key=lambda f: f["confidence"])
            crop = align_and_crop(img, best_face, aligner)

            if crop is None:
                no_face_count += 1
                crop = cv2.resize(img, (224, 224))
                emb = embed(embedder_session, input_name, crop)
                results.append({
                    "identity": identity,
                    "image_path": str(img_path),
                    "embedding": emb,
                    "face_detected": False,
                    "face_count": len(faces),
                })
                continue

            emb = embed(embedder_session, input_name, crop)
            results.append({
                "identity": identity,
                "image_path": str(img_path),
                "embedding": emb,
                "face_detected": True,
                "face_count": len(faces),
            })

    print(f"\nProcessed {total_images} images:")
    print(f"  Face detected: {total_images - no_face_count}/{total_images} "
          f"({100*(total_images-no_face_count)/total_images:.1f}%)")
    print(f"  Multi-face images: {multi_face_count}")
    print(f"  No face (fallback to full image): {no_face_count}")

    return results


def evaluate_retrieval(results):
    """Compute retrieval metrics using identity as ground truth."""
    embeddings = np.array([r["embedding"] for r in results])
    identities = [r["identity"] for r in results]
    unique_ids = sorted(set(identities))
    id_to_label = {name: i for i, name in enumerate(unique_ids)}
    labels = np.array([id_to_label[r["identity"]] for r in results])

    N = len(embeddings)

    # Cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, -1.0)

    # Recall@k
    ks = [1, 3, 5]
    recall = {k: 0.0 for k in ks}
    mrr_sum = 0.0

    for i in range(N):
        ranked = np.argsort(-sim_matrix[i])
        ranked_labels = labels[ranked]
        query_label = labels[i]

        for k in ks:
            if np.any(ranked_labels[:k] == query_label):
                recall[k] += 1.0

        correct_mask = ranked_labels == query_label
        if correct_mask.any():
            first_hit = np.argmax(correct_mask) + 1
            mrr_sum += 1.0 / first_hit

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall[k] / N
    metrics["MRR"] = mrr_sum / N

    # Balanced pair metrics (same approach as fixed ranking_metrics)
    label_to_idx = defaultdict(list)
    for i, l in enumerate(labels):
        label_to_idx[l].append(i)
    valid_labels = [l for l in label_to_idx if len(label_to_idx[l]) >= 2]

    rng = np.random.RandomState(42)
    num_pairs = min(5000, len(valid_labels) * 10)
    half = num_pairs // 2

    pos_sims, neg_sims = [], []

    for _ in range(half):
        l = rng.choice(valid_labels)
        i, j = rng.choice(label_to_idx[l], size=2, replace=False)
        pos_sims.append(sim_matrix[i, j])

    all_labels = list(label_to_idx.keys())
    for _ in range(half):
        l1, l2 = rng.choice(len(all_labels), size=2, replace=False)
        i = rng.choice(label_to_idx[all_labels[l1]])
        j = rng.choice(label_to_idx[all_labels[l2]])
        neg_sims.append(sim_matrix[i, j])

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    metrics["pos_sim_mean"] = float(np.mean(pos_sims))
    metrics["pos_sim_std"] = float(np.std(pos_sims))
    metrics["neg_sim_mean"] = float(np.mean(neg_sims))
    metrics["neg_sim_std"] = float(np.std(neg_sims))

    # Find best threshold and accuracy
    all_sims = np.concatenate([pos_sims, neg_sims])
    all_labels_binary = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])

    best_acc = 0
    best_thresh = 0
    for t in np.arange(-0.3, 1.0, 0.01):
        preds = (all_sims >= t).astype(int)
        acc = (preds == all_labels_binary).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    metrics["verification_accuracy"] = float(best_acc)
    metrics["best_threshold"] = float(best_thresh)

    # Per-identity retrieval breakdown
    per_identity = {}
    for l in valid_labels:
        indices = label_to_idx[l]
        identity_name = [r["identity"] for r in [results[i] for i in indices]][0]
        # For each image of this identity, check if R@1 is correct
        hits = 0
        for idx in indices:
            ranked = np.argsort(-sim_matrix[idx])
            if labels[ranked[0]] == l:
                hits += 1
        per_identity[identity_name] = {
            "num_images": len(indices),
            "recall@1": hits / len(indices),
        }

    metrics["per_identity"] = per_identity

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Battersea Pets")
    parser.add_argument("--species", choices=["cat", "dog"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/root/battersea-pets"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cat-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_cat_v2/pet_embedder.onnx"))
    parser.add_argument("--dog-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_dog_v2/pet_embedder.onnx"))
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(f"/root/pet_embedding/eval_battersea_{args.species}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load face detection model
    print("Loading face detection model...")
    sys.path.insert(0, "/root/pipeline/yolov5_face")
    face_model = load_face_model(DEFAULT_FACE_MODEL, device=device, cleanup_aliases=False)
    face_model = face_model.to(device)

    # Load face aligner
    aligner = FaceAligner()

    # Load embedding model
    onnx_path = args.cat_model if args.species == "cat" else args.dog_model
    print(f"Loading embedder: {onnx_path}")
    embedder_session, input_name = load_embedder(onnx_path)

    # Process dataset
    print(f"\nProcessing {args.species} images...")
    results = process_dataset(
        args.data_dir, args.species, face_model, aligner,
        embedder_session, input_name, device=device,
    )

    if not results:
        print("No results!")
        return

    # Filter to identities with >= 2 images for meaningful eval
    id_counts = defaultdict(int)
    for r in results:
        id_counts[r["identity"]] += 1
    valid_ids = {k for k, v in id_counts.items() if v >= 2}
    results_filtered = [r for r in results if r["identity"] in valid_ids]
    excluded = len(results) - len(results_filtered)
    if excluded > 0:
        print(f"\nExcluded {excluded} images from {len(id_counts) - len(valid_ids)} "
              f"identities with < 2 images")

    print(f"\nEvaluating on {len(results_filtered)} images, "
          f"{len(valid_ids)} identities...")

    # Evaluate
    metrics = evaluate_retrieval(results_filtered)

    # Print results
    print("\n" + "=" * 60)
    print(f"BATTERSEA {args.species.upper()} — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Images:               {len(results_filtered)}")
    print(f"  Identities:           {len(valid_ids)}")
    face_rate = sum(1 for r in results_filtered if r["face_detected"]) / len(results_filtered)
    print(f"  Face detection rate:  {face_rate:.1%}")
    print(f"  Recall@1:             {metrics['recall@1']:.3f}")
    print(f"  Recall@3:             {metrics['recall@3']:.3f}")
    print(f"  Recall@5:             {metrics['recall@5']:.3f}")
    print(f"  MRR:                  {metrics['MRR']:.3f}")
    print(f"  Verification acc:     {metrics['verification_accuracy']:.3f}")
    print(f"  Best threshold:       {metrics['best_threshold']:.2f}")
    print(f"  Pos sim (mean±std):   {metrics['pos_sim_mean']:.3f} ± {metrics['pos_sim_std']:.3f}")
    print(f"  Neg sim (mean±std):   {metrics['neg_sim_mean']:.3f} ± {metrics['neg_sim_std']:.3f}")

    # Per-identity breakdown
    per_id = metrics.pop("per_identity")
    print(f"\n  Per-identity R@1 (worst 10):")
    sorted_ids = sorted(per_id.items(), key=lambda x: x[1]["recall@1"])
    for name, info in sorted_ids[:10]:
        print(f"    {name:<30} R@1={info['recall@1']:.2f}  ({info['num_images']} imgs)")

    print(f"\n  Per-identity R@1 (best 10):")
    for name, info in sorted_ids[-10:]:
        print(f"    {name:<30} R@1={info['recall@1']:.2f}  ({info['num_images']} imgs)")

    # Save results
    save_metrics = {k: v for k, v in metrics.items()}
    save_metrics["per_identity"] = per_id
    save_metrics["species"] = args.species
    save_metrics["num_images"] = len(results_filtered)
    save_metrics["num_identities"] = len(valid_ids)
    save_metrics["face_detection_rate"] = face_rate

    with open(args.output_dir / "battersea_results.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"\nSaved results to {args.output_dir / 'battersea_results.json'}")


if __name__ == "__main__":
    main()
