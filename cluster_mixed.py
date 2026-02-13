#!/usr/bin/env python3
"""Cluster mixed cat+dog images from a single flat directory.

Detects faces with YOLOv5 (which outputs class: 0=cat, 1=dog),
embeds with species-specific ONNX models using multi-crop TTA,
clusters each species separately with HDBSCAN, and saves results.

Usage:
    python cluster_mixed.py
    python cluster_mixed.py --no-tta          # disable TTA for comparison
    python cluster_mixed.py --tta-crops 5     # number of TTA crops
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
sys.path.insert(0, "/root/pipeline/yolov5_face")

from face_loader import load_face_model, DEFAULT_FACE_MODEL
from face_align import FaceAligner

import onnxruntime as ort
import hdbscan
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "cat", 1: "dog"}


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


def embed_single(session, input_name, face_bgr):
    inp = preprocess(face_bgr)
    embedding = session.run(None, {input_name: inp})[0][0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def generate_tta_crops(face_bgr, num_crops=5):
    """Generate augmented views of a face crop for TTA.

    Crops generated:
      0: original
      1: horizontal flip
      2: slight zoom (center crop 90%)
      3: slight zoom + flip
      4+: small random rotations (-10, +10 degrees)
    """
    h, w = face_bgr.shape[:2]
    crops = [face_bgr]  # original

    # 1: horizontal flip
    crops.append(cv2.flip(face_bgr, 1))

    if num_crops >= 3:
        # 2: center crop 90%
        margin = int(min(h, w) * 0.05)
        center_crop = face_bgr[margin:h - margin, margin:w - margin]
        center_crop = cv2.resize(center_crop, (w, h))
        crops.append(center_crop)

    if num_crops >= 4:
        # 3: center crop + flip
        crops.append(cv2.flip(center_crop, 1))

    if num_crops >= 5:
        # 4: slight rotation -10
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -10, 1.0)
        rotated = cv2.warpAffine(face_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        crops.append(rotated)

    if num_crops >= 6:
        # 5: slight rotation +10
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
        rotated = cv2.warpAffine(face_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        crops.append(rotated)

    if num_crops >= 7:
        # 6: brightness jitter (darker)
        darker = np.clip(face_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        crops.append(darker)

    if num_crops >= 8:
        # 7: brightness jitter (brighter)
        brighter = np.clip(face_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        crops.append(brighter)

    return crops[:num_crops]


def embed_tta(session, input_name, face_bgr, num_crops=5):
    """Embed with multi-crop TTA: average embeddings from augmented views."""
    crops = generate_tta_crops(face_bgr, num_crops)

    embeddings = []
    for crop in crops:
        emb = embed_single(session, input_name, crop)
        embeddings.append(emb)

    # Average and re-normalize
    avg_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm
    return avg_emb


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
            class_id = int(d[-1]) if len(d) > 5 else -1
            faces.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "landmarks": landmarks,
                "class_id": class_id,
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


def create_mixed_dir(battersea_dir, mixed_dir):
    """Dump all cat+dog images from battersea-pets into a single flat directory."""
    mixed_dir = Path(mixed_dir)
    if mixed_dir.exists():
        shutil.rmtree(mixed_dir)
    mixed_dir.mkdir(parents=True)

    battersea_dir = Path(battersea_dir)
    ground_truth = {}
    count = 0

    for species in ["cat", "dog"]:
        species_dir = battersea_dir / species
        if not species_dir.exists():
            continue
        for identity_dir in sorted(species_dir.iterdir()):
            if not identity_dir.is_dir() or identity_dir.name.startswith("."):
                continue
            identity = identity_dir.name
            for img_path in sorted(identity_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                flat_name = f"{species}_{identity}_{img_path.name}"
                dst = mixed_dir / flat_name
                shutil.copy2(img_path, dst)
                ground_truth[flat_name] = {
                    "species": species,
                    "identity": identity,
                }
                count += 1

    with open(mixed_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Created mixed directory with {count} images")
    return ground_truth


def process_mixed(mixed_dir, face_model, aligner, embedders, device="cuda",
                  use_tta=True, tta_crops=5):
    """Detect faces, classify species, compute embeddings."""
    mixed_dir = Path(mixed_dir)
    gt_path = mixed_dir / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)
    else:
        ground_truth = {}

    images = sorted([
        f for f in mixed_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    embed_fn_label = f"TTA x{tta_crops}" if use_tta else "single crop"
    print(f"Processing {len(images)} mixed images ({embed_fn_label})...")

    records = {"cat": [], "dog": []}
    species_predictions = {"cat": 0, "dog": 0, "unknown": 0}
    species_correct = 0
    species_total = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = detect_faces_yolo(face_model, img, conf_thresh=0.3, device=device)

        gt_info = ground_truth.get(img_path.name, {})
        gt_species = gt_info.get("species", "unknown")
        gt_identity = gt_info.get("identity", img_path.stem)

        if len(faces) == 0:
            species_predictions["unknown"] += 1
            continue

        best_face = max(faces, key=lambda f: f["confidence"])
        class_id = best_face.get("class_id", -1)

        if class_id == 0:
            pred_species = "cat"
        elif class_id == 1:
            pred_species = "dog"
        else:
            species_predictions["unknown"] += 1
            continue

        species_predictions[pred_species] += 1
        if gt_species != "unknown":
            species_total += 1
            if pred_species == gt_species:
                species_correct += 1

        crop = align_and_crop(img, best_face, aligner)
        if crop is None:
            crop = cv2.resize(img, (224, 224))

        session, input_name = embedders[pred_species]
        if use_tta:
            emb = embed_tta(session, input_name, crop, num_crops=tta_crops)
        else:
            emb = embed_single(session, input_name, crop)

        records[pred_species].append({
            "identity": gt_identity,
            "gt_species": gt_species,
            "pred_species": pred_species,
            "image_path": str(img_path),
            "embedding": emb,
        })

    print(f"\nSpecies detection results:")
    print(f"  Cat: {species_predictions['cat']}")
    print(f"  Dog: {species_predictions['dog']}")
    print(f"  Unknown (no face): {species_predictions['unknown']}")
    if species_total > 0:
        acc = species_correct / species_total * 100
        print(f"  Species accuracy: {species_correct}/{species_total} = {acc:.1f}%")

    return records, species_predictions, (species_correct, species_total)


def cluster_species(records, output_dir, species, min_cluster_size=2, min_samples=2):
    """Cluster one species with HDBSCAN."""
    if len(records) < 2:
        print(f"\n  {species}: too few images ({len(records)}), skipping clustering")
        return None

    embeddings = np.array([r["embedding"] for r in records])
    gt_identities = [r["identity"] for r in records]

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

    species_dir = output_dir / species
    if species_dir.exists():
        shutil.rmtree(species_dir)
    species_dir.mkdir(parents=True)

    for i, record in enumerate(records):
        cluster_id = pred_labels[i]
        if cluster_id == -1:
            cluster_dir = species_dir / "unclustered"
        else:
            cluster_dir = species_dir / f"cluster_{cluster_id:03d}"

        cluster_dir.mkdir(exist_ok=True)
        src = Path(record["image_path"])
        dst_name = f"{record['identity']}__{src.name}"
        dst = cluster_dir / dst_name
        shutil.copy2(src, dst)

    mask = pred_labels != -1
    if mask.sum() > 0:
        gt_labels_str = [gt_identities[i] for i in range(len(records)) if mask[i]]
        unique_gt = sorted(set(gt_labels_str))
        gt_map = {name: idx for idx, name in enumerate(unique_gt)}
        gt_numeric = [gt_map[g] for g in gt_labels_str]
        pred_numeric = pred_labels[mask]

        nmi = normalized_mutual_info_score(gt_numeric, pred_numeric)
        ari = adjusted_rand_score(gt_numeric, pred_numeric)

        cluster_ids_set = set(pred_numeric)
        total_correct = 0
        for c in cluster_ids_set:
            c_mask = pred_numeric == c
            c_gt = [gt_numeric[i] for i in range(len(gt_numeric)) if c_mask[i]]
            most_common = max(set(c_gt), key=c_gt.count)
            total_correct += c_gt.count(most_common)
        purity = total_correct / len(gt_numeric)
    else:
        nmi = ari = purity = 0.0

    cluster_contents = defaultdict(lambda: defaultdict(int))
    for i, record in enumerate(records):
        c = pred_labels[i]
        label = "unclustered" if c == -1 else f"cluster_{c:03d}"
        cluster_contents[label][record["identity"]] += 1

    misclassified = sum(1 for r in records if r["gt_species"] != species and r["gt_species"] != "unknown")

    return {
        "species": species,
        "n_images": len(records),
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "n_clustered": int(mask.sum()),
        "misclassified_species": misclassified,
        "NMI": float(nmi),
        "ARI": float(ari),
        "purity": float(purity),
        "cluster_contents": {k: dict(v) for k, v in sorted(cluster_contents.items())},
    }


def print_results(metrics):
    species = metrics["species"].upper()
    print(f"\n{'=' * 60}")
    print(f"  {species} CLUSTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total images:    {metrics['n_images']}")
    print(f"  Clusters found:  {metrics['n_clusters']}")
    print(f"  Clustered:       {metrics['n_clustered']}")
    print(f"  Unclustered:     {metrics['n_noise']}")
    if metrics.get('misclassified_species', 0) > 0:
        print(f"  Misclassified:   {metrics['misclassified_species']} (wrong species)")
    print(f"  NMI:             {metrics['NMI']:.3f}")
    print(f"  ARI:             {metrics['ARI']:.3f}")
    print(f"  Purity:          {metrics['purity']:.3f}")

    print(f"\n  Cluster breakdown:")
    for cluster_name, identities in sorted(metrics["cluster_contents"].items()):
        total = sum(identities.values())
        id_str = ", ".join(f"{k}({v})" for k, v in sorted(identities.items(), key=lambda x: -x[1]))
        pure = "PURE" if len(identities) == 1 else "MIXED"
        print(f"    {cluster_name:<20} [{pure:5s}] {total:3d} imgs — {id_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battersea-dir", type=Path, default=Path("/root/battersea-pets"))
    parser.add_argument("--mixed-dir", type=Path, default=Path("/root/pet_embedding/battersea_mixed"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/pet_embedding/battersea_mixed_clusters"))
    parser.add_argument("--cat-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_cat_v2/pet_embedder.onnx"))
    parser.add_argument("--dog-model", type=Path,
                        default=Path("/root/pet_embedding/outputs_dog_v2/pet_embedder.onnx"))
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--skip-dump", action="store_true",
                        help="Skip creating mixed dir (reuse existing)")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--tta-crops", type=int, default=5,
                        help="Number of TTA crops (default: 5)")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    use_tta = not args.no_tta

    # Step 1: Dump all images into a single flat directory
    if not args.skip_dump:
        print("\n--- Step 1: Creating mixed image directory ---")
        ground_truth = create_mixed_dir(args.battersea_dir, args.mixed_dir)
    else:
        print("\n--- Step 1: Using existing mixed directory ---")

    # Step 2: Load models
    print("\n--- Step 2: Loading models ---")
    print("Loading face detection model...")
    face_model = load_face_model(DEFAULT_FACE_MODEL, device=device, cleanup_aliases=False)
    face_model = face_model.to(device)
    aligner = FaceAligner()

    print(f"Loading cat embedder: {args.cat_model}")
    cat_session, cat_input = load_embedder(args.cat_model)
    print(f"Loading dog embedder: {args.dog_model}")
    dog_session, dog_input = load_embedder(args.dog_model)

    embedders = {
        "cat": (cat_session, cat_input),
        "dog": (dog_session, dog_input),
    }

    # Step 3: Detect, classify species, embed
    print("\n--- Step 3: Detect faces, classify species, embed ---")
    records, species_counts, (correct, total) = process_mixed(
        args.mixed_dir, face_model, aligner, embedders, device=device,
        use_tta=use_tta, tta_crops=args.tta_crops,
    )

    # Step 4: Cluster each species separately
    print("\n--- Step 4: Clustering ---")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_metrics = {}
    for species in ["cat", "dog"]:
        print(f"\nClustering {species} ({len(records[species])} images)...")
        metrics = cluster_species(
            records[species], output_dir, species,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        if metrics:
            all_metrics[species] = metrics
            print_results(metrics)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 60}")
    tta_label = f"TTA x{args.tta_crops}" if use_tta else "No TTA"
    print(f"  Embedding mode:          {tta_label}")
    print(f"  Species detection acc:   {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Cat images detected:     {species_counts['cat']}")
    print(f"  Dog images detected:     {species_counts['dog']}")
    print(f"  No face detected:        {species_counts['unknown']}")
    for sp, m in all_metrics.items():
        print(f"\n  {sp.upper()}: {m['n_clusters']} clusters, "
              f"Clustered={m['n_clustered']}/{m['n_images']} ({m['n_clustered']/m['n_images']*100:.0f}%), "
              f"NMI={m['NMI']:.3f}, ARI={m['ARI']:.3f}, Purity={m['purity']:.3f}")

    # Save all metrics
    summary = {
        "tta": use_tta,
        "tta_crops": args.tta_crops if use_tta else 0,
        "species_detection": {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "counts": species_counts,
        },
        "clustering": all_metrics,
    }
    with open(output_dir / "mixed_cluster_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
