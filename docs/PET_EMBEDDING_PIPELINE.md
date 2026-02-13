# Pet Face Embedding Pipeline

## Complete Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation Pipeline](#6-evaluation-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Export and Deployment](#8-export-and-deployment)
9. [Results Summary](#9-results-summary)
10. [File Reference](#10-file-reference)
11. [Environment Setup](#11-environment-setup)

---

## 1. Project Overview

### Goal

Build a lightweight pet face recognition system that can:
- Generate a compact **128-dimensional embedding** for any pet face image
- Determine if two images show the **same individual pet** via cosine similarity
- Run efficiently on **mobile devices** (< 2 MB model, < 5 ms inference)

### Approach

The system uses **ArcFace metric learning** to train an embedding model. During
training, the model learns to map images of the same pet close together in
embedding space and images of different pets far apart. At inference time, only
the embedding backbone is used — the ArcFace classification head is discarded.

### Supported Backbone Architectures

| Architecture | Pretrained | Params (embedder) | Best For |
|-------------|-----------|-------------------|----------|
| MobileNetV3-Small + GDConv | Yes (ImageNet) | 1,701,408 | Maximum accuracy |
| MobileFaceNet | No (from scratch) | 1,082,624 | Robustness, small size |

---

## 2. System Architecture

### High-Level Flow

```
Input Image
    |
    v
[Face Detection] -----> Bounding box (YOLOv5 or similar)
    |
    v
[Crop + Align] -------> 224x224 RGB face image
    |
    v
[Preprocessing] -------> Normalized tensor (ImageNet mean/std)
    |
    v
[Embedding Model] -----> 128-dim L2-normalized vector
    |
    v
[Matching] ------------> Cosine similarity vs gallery
    |
    v
Identity + Confidence
```

### Training vs Inference

```
TRAINING:
  Image -> Augment -> Backbone -> Head -> 128-dim embedding -> ArcFace -> Cross-Entropy Loss
                                                                  |
                                                          (class logits for
                                                           N identities)

INFERENCE:
  Image -> Preprocess -> Backbone -> Head -> 128-dim embedding -> Cosine Similarity
                                                                      |
                                                              (match against gallery)
```

The ArcFace head (which has N_identities x 128 parameters) is **discarded** after
training. Only the embedding backbone + head (~1-1.7M params) is exported.

---

## 3. Data Pipeline

### 3.1 Dataset Structure

Data is organized as **one folder per identity**, with each folder containing
multiple aligned face images of the same individual pet.

```
cat_aligned/
  14210250/
    0_face0.jpg
    1_face0.jpg
    2_face0.jpg
  15499650/
    0_face0.jpg
    1_face0.jpg
    ...
  ...

dog_aligned/
  <identity_id>/
    <image_id>_face<N>.jpg
  ...
```

### 3.2 Dataset Statistics

| | Cat | Dog |
|---|---|---|
| Identities | 19,957 | 22,826 |
| Total images | 86,402 | 98,643 |
| Images per identity (mean) | 4.3 | 4.3 |
| Images per identity (median) | 4 | 4 |
| Images per identity (min / max) | 3 / 22 | 3 / 19 |
| Images per identity (std) | 1.6 | 1.4 |

### 3.3 Data Splits

Splitting is done **by identity** (not by image) to ensure no identity leaks
across splits:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 80% | ArcFace classification training |
| Validation | 10% | Verification accuracy (pair matching) |
| Test | 10% | Full benchmark evaluation |

The split is deterministic (seed=42) and shuffles identity folders before
splitting, ensuring reproducibility.

**Implementation**: `dataset.py` → `load_dataset()`

### 3.4 Data Augmentation

Training images undergo extensive augmentation to improve generalization.
Validation and test images use only resize + normalize.

**Training Augmentations** (`dataset.py` → `get_train_transforms()`):

| Category | Augmentation | Parameters | Probability |
|----------|-------------|------------|-------------|
| **Geometric** | Rotation | limit=25 degrees | 0.5 |
| | Horizontal flip | — | 0.5 |
| | Affine (translate + scale) | translate=10%, scale=0.85-1.15 | 0.3 |
| **Color** | Color jitter | brightness/contrast/sat=0.3, hue=0.1 | 0.5 |
| | Grayscale | — | 0.05 |
| **Quality** | Gaussian blur | kernel 3-9 | 0.2 |
| | Gaussian noise | std=0.01-0.05 | 0.2 |
| | JPEG compression | quality=70-100 | 0.2 |
| | Downscale | scale=0.4-0.8 | 0.15 |
| **Occlusion** | Random shadow | — | 0.1 |
| | Coarse dropout | 1-4 holes, 16-48px | 0.3 |
| **Normalize** | ImageNet normalize | mean=[0.485,0.456,0.406] | 1.0 |

**Validation/Test Transforms** (`dataset.py` → `get_val_transforms()`):

| Step | Parameters |
|------|------------|
| Resize | 224x224 |
| Normalize | ImageNet mean/std |

### 3.5 Pair Dataset for Verification

For validation during training, the system generates **verification pairs**:

- **Positive pairs**: Two different images of the same identity
- **Negative pairs**: Images from two different identities
- 50/50 split (2,500 positive + 2,500 negative = 5,000 pairs)

The model computes embeddings for both images and measures cosine similarity.
The best threshold is found by sweeping from -0.5 to 1.0 in 0.01 increments.

**Implementation**: `dataset.py` → `PairDataset`

---

## 4. Model Architecture

### 4.1 Overall Structure

```
PetFaceModel (training only)
  |-- PetFaceEmbedder (kept for inference)
  |     |-- backbone (feature extractor)
  |     |-- embedding head (projection to 128-dim)
  |
  |-- ArcFaceHead (discarded after training)
        |-- weight: (num_classes, 128) normalized classifier
```

### 4.2 Backbone Options

#### MobileNetV3-Small (timm)

- Pretrained on ImageNet-1K
- Used with `global_pool=""` to output feature maps (not pooled features)
- Feature map output: (B, 576, 7, 7) at 224x224 input
- Modified with `drop_rate=0.2`

#### MobileFaceNet

Custom architecture based on the MobileFaceNet paper, designed specifically for
face recognition:

```
Input (3, 224, 224)
    |
Conv2d 3x3/2 -> 64 channels          # (64, 112, 112)
DepthwiseSeparable -> 64 channels     # (64, 112, 112)
    |
Bottleneck x5 (exp=2, out=64, /2)    # (64, 56, 56)
Bottleneck x1 (exp=4, out=128, /2)   # (128, 28, 28)
Bottleneck x6 (exp=2, out=128, /1)   # (128, 28, 28)
Bottleneck x1 (exp=4, out=128, /2)   # (128, 14, 14)
Bottleneck x2 (exp=2, out=128, /1)   # (128, 14, 14)
    |
Conv1x1 -> 512 channels              # (512, 14, 14)
GDConv 14x14                         # (512, 1, 1)
Conv1x1 -> 128 channels              # (128, 1, 1)
Dropout(0.4)
Flatten + L2 Normalize               # (128,)
```

**Bottleneck block** = inverted residual (MobileNetV2-style):
Expansion (1x1) -> Depthwise (3x3) -> Projection (1x1), with residual if
stride=1 and channels match.

**GDConv** (Global Depthwise Convolution): A depthwise convolution with kernel
size equal to the feature map size. Unlike global average pooling, it learns
position-aware weights, preserving spatial information important for recognition.

### 4.3 Embedding Head

#### GDConv Head (used with MobileNetV3)

```
Feature maps (B, C, H, W)
    |
GDConv (depthwise conv, kernel=H)  -> (B, C, 1, 1)
Dropout(0.4)
Conv1x1 -> embedding_dim            -> (B, 128, 1, 1)
BatchNorm2d
Flatten                              -> (B, 128)
L2 Normalize                         -> (B, 128) unit vectors
```

#### MLP Head (alternative, not used in final models)

```
Pooled features (B, C)
    |
Dropout(0.4) -> Linear(C, 512) -> BN -> PReLU -> Dropout(0.2) -> Linear(512, 128) -> BN
L2 Normalize -> (B, 128) unit vectors
```

### 4.4 ArcFace Head

ArcFace (Additive Angular Margin Loss) is the training objective. It adds an
angular margin penalty to the target class, forcing the model to learn more
discriminative embeddings.

```
Input: embeddings (B, 128), labels (B,)

1. Normalize embeddings and weight matrix
2. Compute cosine similarity: cos(theta) = W_norm @ emb_norm
3. Convert to angle: theta = arccos(cos_theta)
4. Add margin to target class: theta_target = theta + margin
5. Convert back: logit_target = cos(theta + margin)
6. Scale: output = logits * scale
7. Apply CrossEntropyLoss with label smoothing (0.1)
```

**Parameters**:
- Scale (s): 32.0 — controls the temperature of the softmax
- Margin (m): 0.4 — angular margin in radians (~23 degrees)
- Margin warmup: linearly increases from 0 to 0.4 over the first 1/3 of training

**Implementation**: `model.py` → `ArcFaceHead`

---

## 5. Training Pipeline

### 5.1 Training Configuration

| Parameter | MobileNetV3-Small | MobileFaceNet |
|-----------|------------------|---------------|
| Optimizer | AdamW | AdamW |
| Base LR | 5e-4 | 1e-3 |
| Backbone LR | 5e-5 (0.1x) | 1e-3 (1.0x) |
| Head LR | 5e-4 (1.0x) | 1e-3 (1.0x) |
| Weight decay | 5e-4 | 5e-4 |
| Epochs | 40 | 60 |
| Batch size | 96 | 96 |
| LR schedule | Cosine annealing (eta_min=1e-6) | Same |
| LR warmup | 5 epochs, linear | Same |
| ArcFace scale | 32 | 32 |
| ArcFace margin | 0.4 | 0.4 |
| Margin warmup | First 1/3 of epochs | Same |
| Label smoothing | 0.1 | 0.1 |
| Gradient clipping | max_norm=1.0 | Same |
| Image size | 224x224 | 224x224 |
| Embedding dim | 128 | 128 |
| Dropout | 0.4 | 0.4 |

**Key differences**:
- MobileNetV3 uses **0.1x LR for backbone** (pretrained, avoid destroying
  ImageNet features)
- MobileFaceNet uses **full LR for backbone** (training from scratch)
- MobileFaceNet trains for **60 epochs** (no pretrain = more training needed)

### 5.2 Learning Rate Schedule

```
LR
 ^
 |   /----\
 |  /      \
 | /        \
 |/          \__________
 +---|-------|----------> epochs
   warmup  cosine annealing
   (5 ep)   (remaining)
```

1. **Warmup** (epochs 0-4): Linear ramp from 0 to base_lr
2. **Cosine annealing** (epochs 5+): Cosine decay from base_lr to 1e-6

### 5.3 Margin Warmup Schedule

```
Margin
 ^
 |          ___________
 |         /
 |        /
 |       /
 |      /
 |_____/
 +-----|---------------> epochs
    1/3 of total
```

Starting with margin=0 allows the model to first learn basic features, then
gradually increase the angular margin to enforce tighter clustering. This is
critical for training from scratch — without warmup, the loss becomes too hard
and the model fails to converge.

### 5.4 Validation During Training

Every 3 epochs (and the final epoch), the model is evaluated on the validation
split using **verification accuracy**:

1. Generate 5,000 pairs (2,500 positive, 2,500 negative) from val identities
2. Compute embeddings for all pairs
3. Compute cosine similarity
4. Sweep thresholds to find best accuracy
5. Report accuracy, threshold, precision, recall, F1

Best model is saved when val accuracy improves.

### 5.5 Training Commands

```bash
# MobileNetV3-Small
python train.py --data-dir ~/cat_aligned --output-dir outputs_cat \
    --backbone mobilenetv3_small_100 --epochs 40 --batch-size 96 --lr 5e-4

python train.py --data-dir ~/dog_aligned --output-dir outputs_dog \
    --backbone mobilenetv3_small_100 --epochs 40 --batch-size 96 --lr 5e-4

# MobileFaceNet
python train.py --data-dir ~/cat_aligned --output-dir outputs_cat_mfn \
    --backbone mobilefacenet --epochs 60 --batch-size 96 --lr 1e-3

python train.py --data-dir ~/dog_aligned --output-dir outputs_dog_mfn \
    --backbone mobilefacenet --epochs 60 --batch-size 96 --lr 1e-3
```

### 5.6 Training Outputs

Each training run produces:

| File | Description |
|------|-------------|
| `config.json` | Training hyperparameters |
| `best_model.pt` | Checkpoint with best val accuracy |
| `last_checkpoint.pt` | Most recent checkpoint |
| `history.json` | Per-epoch loss, LR, margin, val metrics |
| `pet_embedder.pt` | Extracted embedder weights |
| `pet_embedder.onnx` | ONNX export for deployment |

**Implementation**: `train.py`

---

## 6. Evaluation Pipeline

### 6.1 Overview

The evaluation pipeline runs 7 benchmark categories on the held-out test split.
It automatically detects the model architecture from `config.json`.

### 6.2 Benchmarks

#### A. Retrieval

Computes a full cosine similarity matrix on all test embeddings and evaluates
ranked retrieval quality.

| Metric | Description |
|--------|-------------|
| Recall@1 | % of queries where the top-1 result is correct |
| Recall@5 | % where at least one of top-5 is correct |
| Recall@10 | % where at least one of top-10 is correct |
| MRR | Mean Reciprocal Rank — average of 1/rank of first correct result |
| NDCG@10 | Normalized Discounted Cumulative Gain at rank 10 |

#### B. Clustering

Runs KMeans with k = number of test identities and compares predicted clusters
to true identity labels.

| Metric | Description |
|--------|-------------|
| NMI | Normalized Mutual Information (0-1, higher = better) |
| ARI | Adjusted Rand Index (-1 to 1, higher = better) |
| Purity | Fraction of samples in each cluster belonging to majority class |

#### C. Similarity Ranking

Samples 50,000 random pairs and evaluates binary classification (same vs
different) using cosine similarity as the score.

| Metric | Description |
|--------|-------------|
| ROC-AUC | Area under the ROC curve |
| EER | Equal Error Rate (where FPR = FNR) |
| Spearman | Rank correlation between similarity and label |

#### D. Triplet Metrics

Samples 20,000 (anchor, positive, negative) triplets and measures how well the
model respects triplet constraints.

| Metric | Description |
|--------|-------------|
| Violation rate | % of triplets where negative is closer than positive |
| Average margin | Mean of (sim_positive - sim_negative) |

#### E. Overfitting Check

Compares Recall@1 on 2,000 sampled train images vs 2,000 test images. A large
gap (train >> test) indicates overfitting.

#### F. Robustness Probes

Applies 5 degradations to 1,000 test images and measures:
- **Similarity drop**: 1 - cosine_similarity(clean, degraded)
- **Recall@1 drop**: clean_recall - degraded_recall

| Degradation | Parameters |
|-------------|------------|
| Gaussian blur | radius=3.5 |
| Gaussian noise | std=0.05 |
| JPEG compression | quality=30 |
| Brightness shift | 1.5x multiplier |
| Partial occlusion | 25% center black rectangle |

#### G. Visualizations

| Output | Description |
|--------|-------------|
| `nearest_neighbors.png` | Grid of 10 queries with top-5 nearest neighbors (green=correct, red=wrong) |
| `similarity_distribution.png` | Histogram of positive vs negative pair similarities |
| `roc_curve.png` | ROC curve with AUC |
| `robustness.png` | Bar chart of similarity/recall drop per degradation |

### 6.3 Evaluation Commands

```bash
python evaluate_embeddings.py \
    --model-path outputs_cat/best_model.pt \
    --data-dir ~/cat_aligned \
    --output-dir eval_cat \
    --image-size 224

# Optional: enable test-time augmentation (flip averaging)
python evaluate_embeddings.py \
    --model-path outputs_cat/best_model.pt \
    --data-dir ~/cat_aligned \
    --output-dir eval_cat_tta \
    --image-size 224 \
    --tta
```

### 6.4 Model Comparison

```bash
python compare_models.py \
    --models "MNv3-Cat:eval_cat" "MFN-Cat:eval_cat_mfn" \
             "MNv3-Dog:eval_dog" "MFN-Dog:eval_dog_mfn" \
    --output comparison.txt
```

**Implementation**: `evaluate_embeddings.py`, `compare_models.py`

---

## 7. Inference Pipeline

### 7.1 Full Pipeline (Detection + Embedding + Matching)

The `PetRecognitionPipeline` class in `inference.py` provides end-to-end
recognition:

```python
from inference import PetRecognitionPipeline

pipeline = PetRecognitionPipeline(
    detector_path="yolov5_pet_face.pt",
    embedder_path="outputs_cat/pet_embedder.pt",
    image_size=224,
)

# Register pets into gallery
pipeline.register_pet("Whiskers", ["whiskers_1.jpg", "whiskers_2.jpg"])
pipeline.register_pet("Luna", ["luna_1.jpg", "luna_2.jpg"])
pipeline.save_gallery("my_cats.pt")

# Recognize
results = pipeline.recognize("test_photo.jpg", threshold=0.5)
for r in results:
    print(f"{r['identity']} (similarity: {r['similarity']:.2f})")

# Compare two images directly
same, score = pipeline.compare_faces("photo_a.jpg", "photo_b.jpg")
```

### 7.2 ONNX Inference (for deployment)

```python
from inference import ONNXPetRecognition

model = ONNXPetRecognition(
    onnx_path="outputs_cat/pet_embedder.onnx",
    image_size=224,
)

# Get embedding
embedding = model.get_embedding(face_image)  # PIL Image -> np.ndarray (128,)

# Compare two faces
similarity = model.compare(face1, face2)  # float [-1, 1]
```

### 7.3 Preprocessing Requirements

All inference requires consistent preprocessing:

1. Resize to 224x224
2. Convert to float32, scale to [0, 1]
3. Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
4. Channel order: RGB (not BGR)

### 7.4 Similarity Thresholds

Recommended thresholds based on validation data:

| Model | Optimal Threshold | Description |
|-------|-------------------|-------------|
| MNv3 Cat | 0.15-0.20 | Cosine similarity for same identity |
| MNv3 Dog | 0.18-0.22 | |
| MFN Cat | 0.15-0.17 | |
| MFN Dog | 0.17-0.20 | |

Higher threshold = fewer false positives, more false negatives.

**Implementation**: `inference.py`

---

## 8. Export and Deployment

### 8.1 Supported Export Formats

| Format | Use Case | File Size |
|--------|----------|-----------|
| ONNX | Cross-platform, ONNX Runtime | ~276-308 KB |
| CoreML | iOS (via coremltools) | — |
| TorchScript Lite | Android (PyTorch Mobile) | — |
| TFLite | Android/embedded (via onnx2tf) | — |
| Quantized (INT8) | Reduced precision | — |

### 8.2 Auto-Export During Training

The training script automatically exports:
- `pet_embedder.pt` — PyTorch state dict
- `pet_embedder.onnx` — ONNX format (opset 18)

### 8.3 Manual Export

```bash
python export_mobile.py \
    --checkpoint outputs_cat/pet_embedder.pt \
    --output-dir exports/ \
    --image-size 224 \
    --formats onnx coreml torchscript
```

### 8.4 Model Sizes

| Model | Embedder Params | ONNX Size | PyTorch Size |
|-------|----------------|-----------|-------------|
| MobileNetV3 + GDConv | 1,701,408 | 308 KB | ~6.5 MB |
| MobileFaceNet | 1,082,624 | 276 KB | ~4.1 MB |

Note: ONNX files are much smaller than PyTorch checkpoints because they only
contain the inference graph, not optimizer state.

**Implementation**: `export_mobile.py`

---

## 9. Results Summary

### 9.1 Validation Accuracy

| Model | Cat | Dog |
|-------|-----|-----|
| MobileNetV3-Small + GDConv | **92.1%** | **94.6%** |
| MobileFaceNet | 90.7% | 92.8% |

### 9.2 Test Set Evaluation

| Metric | MNv3 Cat | MFN Cat | MNv3 Dog | MFN Dog |
|--------|----------|---------|----------|---------|
| **Retrieval** | | | | |
| Recall@1 | **74.7%** | 67.3% | **79.1%** | 74.0% |
| Recall@5 | **84.6%** | 78.9% | **88.2%** | 84.1% |
| MRR | **0.792** | 0.726 | **0.832** | 0.787 |
| **Ranking** | | | | |
| ROC-AUC | **0.971** | 0.928 | **0.981** | 0.942 |
| EER | **9.6%** | 18.3% | **8.5%** | 12.6% |
| **Clustering** | | | | |
| NMI | **0.909** | 0.895 | **0.918** | 0.908 |
| Purity | **0.695** | 0.643 | **0.722** | 0.682 |
| **Triplet** | | | | |
| Violation rate | **3.5%** | 4.7% | **2.0%** | 3.1% |
| Avg margin | **0.479** | 0.451 | **0.509** | 0.483 |
| **Robustness (sim drop)** | | | | |
| Gaussian blur | 0.379 | **0.222** | 0.365 | **0.159** |
| Gaussian noise | 0.027 | **0.011** | 0.024 | **0.011** |
| JPEG compression | 0.045 | **0.017** | 0.040 | **0.022** |
| Partial occlusion | 0.530 | **0.484** | 0.488 | **0.387** |
| **Efficiency** | | | | |
| Embedder params | 1,701,408 | **1,082,624** | 1,701,408 | **1,082,624** |
| ONNX size | 308 KB | **276 KB** | 308 KB | **276 KB** |
| GPU latency | **2.56 ms** | 2.58 ms | - | - |
| CPU latency | **5.63 ms** | 15.78 ms | - | - |

### 9.3 Key Takeaways

1. **MobileNetV3 is more accurate** across all discrimination metrics, benefiting
   from ImageNet pretraining
2. **MobileFaceNet is significantly more robust** to image degradations (40-60%
   less sensitivity to blur, noise, compression)
3. **Both models are small enough for mobile deployment** (< 310 KB ONNX)
4. **GPU latency is identical** (~2.6 ms); MobileNetV3 is faster on CPU (5.6 vs
   15.8 ms)
5. **Neither model overfits** — both show healthy train/test gaps

---

## 10. File Reference

### Core Files

| File | Lines | Description |
|------|-------|-------------|
| `model.py` | 260 | PetFaceEmbedder, ArcFaceHead, PetFaceModel |
| `mobilefacenet.py` | 257 | MobileFaceNet architecture |
| `dataset.py` | 262 | Data loading, splits, augmentation, pair dataset |
| `config.py` | 44 | Default configuration dataclass |
| `train.py` | 424 | Training loop, LR/margin scheduling, checkpointing |
| `evaluate_embeddings.py` | 792 | Full benchmark suite (7 categories + plots) |
| `compare_models.py` | 240 | Side-by-side model comparison tables |
| `inference.py` | 304 | Recognition pipeline + ONNX inference |
| `export_mobile.py` | 229 | Multi-format export (ONNX, CoreML, TorchScript, TFLite) |

### Output Directories

| Directory | Model | Val Acc | Contents |
|-----------|-------|---------|----------|
| `outputs_cat/` | MNv3 Cat | 92.1% | Checkpoint, ONNX, history |
| `outputs_cat_mfn/` | MFN Cat | 90.7% | Checkpoint, ONNX, history |
| `outputs_dog/` | MNv3 Dog | 94.6% | Checkpoint, ONNX, history |
| `outputs_dog_mfn/` | MFN Dog | 92.8% | Checkpoint, ONNX, history |
| `eval_cat/` | MNv3 Cat | — | Eval results, plots |
| `eval_cat_mfn/` | MFN Cat | — | Eval results, plots |
| `eval_dog/` | MNv3 Dog | — | Eval results, plots |
| `eval_dog_mfn/` | MFN Dog | — | Eval results, plots |

---

## 11. Environment Setup

### Hardware

- **GPU**: NVIDIA RTX 4000 Ada Generation (20 GB VRAM)
- **Server**: root@172.234.228.57

### Software

```
Python 3.11
PyTorch 2.9.1
timm >= 0.9.0
albumentations >= 1.3.0
scikit-learn (for evaluation metrics)
onnx + onnxruntime (for export/inference)
matplotlib (for evaluation plots)
Pillow, numpy, tqdm
```

### Setup

```bash
# Create virtual environment
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install dependencies
pip install torch torchvision timm albumentations Pillow numpy tqdm
pip install onnx onnxruntime scikit-learn matplotlib scipy

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Known Issues

- PyTorch 2.9.1 requires `weights_only=False` when loading checkpoints that
  contain numpy arrays
- ONNX export requires `opset_version=18` with PyTorch 2.9.1
- MobileFaceNet at 224x224 is slower on CPU than expected due to 14x14 GDConv
  kernel; consider 112x112 for CPU deployment
- tqdm progress bars produce binary characters in redirected logs; use
  `cat log | col -b` to strip them
