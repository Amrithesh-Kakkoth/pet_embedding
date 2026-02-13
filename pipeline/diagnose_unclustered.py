#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Diagnose unclustered images: check face detection quality."""

import cv2
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/root/pipeline")
sys.path.insert(0, "/root/pipeline/yolov5_face")

from face_loader import load_face_model, DEFAULT_FACE_MODEL
from face_align import FaceAligner

import onnxruntime as ort
import hdbscan

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_embedder(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session, session.get_inputs()[0].name


def preprocess(bgr_image):
    img = cv2.resize(bgr_image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]


def embed(session, input_name, face_bgr):
    embedding = session.run(None, {input_name: preprocess(face_bgr)})[0][0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def detect_faces_yolo(model, img, conf_thresh=0.3, device="cuda"):
    import torch
    h0, w0 = img.shape[:2]
    scale = 640 / max(h0, w0)
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
            landmarks = d[5:15].reshape(5, 2) / scale if len(d) >= 15 else None
            class_id = int(d[-1]) if len(d) > 5 else -1
            # bbox area relative to image
            bw = (box[2] - box[0])
            bh = (box[3] - box[1])
            bbox_area = bw * bh
            img_area = h0 * w0
            bbox_ratio = bbox_area / img_area
            faces.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "landmarks": landmarks,
                "class_id": class_id,
                "bbox_ratio": float(bbox_ratio),
                "n_landmarks": int(landmarks is not None and np.all(landmarks > 0)),
            })
    return faces


def align_and_crop(img, face, aligner, output_size=224):
    bbox = face["bbox"]
    landmarks = face.get("landmarks")
    if landmarks is not None and len(landmarks) >= 2:
        left_eye = tuple(landmarks[0])
        right_eye = tuple(landmarks[1])
        aligned, info = aligner.align(img, left_eye, right_eye, bbox, output_size)
        if aligned is not None:
            return aligned, "aligned"
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
        return None, "failed"
    return cv2.resize(crop, (output_size, output_size)), "bbox_crop"


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    battersea_dir = Path("/root/battersea-pets")
    cat_model_path = Path("/root/pet_embedding/outputs_cat_v2/pet_embedder.onnx")
    dog_model_path = Path("/root/pet_embedding/outputs_dog_v2/pet_embedder.onnx")

    print("Loading models...")
    face_model = load_face_model(DEFAULT_FACE_MODEL, device=device, cleanup_aliases=False).to(device)
    aligner = FaceAligner()
    cat_session, cat_input = load_embedder(cat_model_path)
    dog_session, dog_input = load_embedder(dog_model_path)
    embedders = {"cat": (cat_session, cat_input), "dog": (dog_session, dog_input)}

    for species in ["cat", "dog"]:
        print(f"\n{'='*60}")
        print(f"  DIAGNOSING {species.upper()}")
        print(f"{'='*60}")

        species_dir = battersea_dir / species
        identity_dirs = sorted([
            d for d in species_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ])

        records = []
        for identity_dir in identity_dirs:
            identity = identity_dir.name
            images = sorted([
                f for f in identity_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                faces = detect_faces_yolo(face_model, img, conf_thresh=0.3, device=device)

                if len(faces) == 0:
                    # No face — use full image resize
                    crop = cv2.resize(img, (224, 224))
                    session, input_name = embedders[species]
                    emb = embed(session, input_name, crop)
                    records.append({
                        "identity": identity,
                        "image_path": str(img_path),
                        "embedding": emb,
                        "face_detected": False,
                        "confidence": 0.0,
                        "bbox_ratio": 0.0,
                        "crop_method": "full_resize",
                        "n_faces": 0,
                    })
                else:
                    best_face = max(faces, key=lambda f: f["confidence"])
                    crop, method = align_and_crop(img, best_face, aligner)
                    if crop is None:
                        crop = cv2.resize(img, (224, 224))
                        method = "full_resize"
                    session, input_name = embedders[species]
                    emb = embed(session, input_name, crop)
                    records.append({
                        "identity": identity,
                        "image_path": str(img_path),
                        "embedding": emb,
                        "face_detected": True,
                        "confidence": best_face["confidence"],
                        "bbox_ratio": best_face["bbox_ratio"],
                        "crop_method": method,
                        "n_faces": len(faces),
                    })

        # Cluster
        embeddings = np.array([r["embedding"] for r in records])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-8)
        cosine_sim = normed @ normed.T
        distance_matrix = np.clip(1.0 - cosine_sim, 0.0, 2.0).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2, min_samples=2,
            metric="precomputed", cluster_selection_method="eom",
        )
        pred_labels = clusterer.fit_predict(distance_matrix)

        clustered_mask = pred_labels != -1
        unclustered_mask = pred_labels == -1

        # --- Stats ---
        print(f"\n  Total: {len(records)}, Clustered: {clustered_mask.sum()}, "
              f"Unclustered: {unclustered_mask.sum()}")

        # Face detection stats
        for label, mask in [("CLUSTERED", clustered_mask), ("UNCLUSTERED", unclustered_mask)]:
            subset = [records[i] for i in range(len(records)) if mask[i]]
            if not subset:
                continue

            n_face = sum(1 for r in subset if r["face_detected"])
            n_no_face = sum(1 for r in subset if not r["face_detected"])
            confs = [r["confidence"] for r in subset if r["face_detected"]]
            bbox_ratios = [r["bbox_ratio"] for r in subset if r["face_detected"]]
            methods = defaultdict(int)
            for r in subset:
                methods[r["crop_method"]] += 1

            print(f"\n  --- {label} ({len(subset)} images) ---")
            print(f"    Face detected:     {n_face}/{len(subset)} ({n_face/len(subset)*100:.1f}%)")
            print(f"    No face detected:  {n_no_face}/{len(subset)} ({n_no_face/len(subset)*100:.1f}%)")
            if confs:
                print(f"    Detection conf:    mean={np.mean(confs):.3f}, "
                      f"median={np.median(confs):.3f}, min={np.min(confs):.3f}, max={np.max(confs):.3f}")
            if bbox_ratios:
                print(f"    Bbox/image ratio:  mean={np.mean(bbox_ratios):.3f}, "
                      f"median={np.median(bbox_ratios):.3f}, min={np.min(bbox_ratios):.3f}")
            print(f"    Crop methods:      {dict(methods)}")

        # Per-identity: how many images per identity, and how many unclustered
        identity_stats = defaultdict(lambda: {"total": 0, "unclustered": 0, "no_face": 0})
        for i, r in enumerate(records):
            identity_stats[r["identity"]]["total"] += 1
            if unclustered_mask[i]:
                identity_stats[r["identity"]]["unclustered"] += 1
            if not r["face_detected"]:
                identity_stats[r["identity"]]["no_face"] += 1

        # Identities that are mostly unclustered
        print(f"\n  Identities with >50% unclustered:")
        for ident, stats in sorted(identity_stats.items()):
            if stats["total"] >= 2 and stats["unclustered"] / stats["total"] > 0.5:
                print(f"    {ident:<25} {stats['unclustered']}/{stats['total']} unclustered, "
                      f"{stats['no_face']} no-face")

        # Identities with only 1 image (can never be clustered)
        singletons = [ident for ident, s in identity_stats.items() if s["total"] == 1]
        print(f"\n  Singleton identities (only 1 image, can never cluster): {len(singletons)}")
        if singletons:
            print(f"    {', '.join(sorted(singletons))}")

        # Check: how many unclustered are from identities with only 1-2 images?
        unclustered_by_identity_size = defaultdict(int)
        for i in range(len(records)):
            if unclustered_mask[i]:
                total_for_identity = identity_stats[records[i]["identity"]]["total"]
                unclustered_by_identity_size[total_for_identity] += 1

        print(f"\n  Unclustered images by identity size:")
        for size in sorted(unclustered_by_identity_size.keys()):
            count = unclustered_by_identity_size[size]
            print(f"    Identities with {size} images: {count} unclustered")

        # Average within-identity similarity for unclustered vs clustered
        print(f"\n  Within-identity similarity (same pet, different images):")
        for label, mask in [("CLUSTERED", clustered_mask), ("UNCLUSTERED", unclustered_mask)]:
            sims = []
            indices = [i for i in range(len(records)) if mask[i]]
            for i in indices:
                for j in indices:
                    if i < j and records[i]["identity"] == records[j]["identity"]:
                        sims.append(float(cosine_sim[i, j]))
            if sims:
                print(f"    {label}: mean={np.mean(sims):.3f}, median={np.median(sims):.3f}, "
                      f"min={np.min(sims):.3f}, n_pairs={len(sims)}")
            else:
                print(f"    {label}: no same-identity pairs")


if __name__ == "__main__":
    main()
