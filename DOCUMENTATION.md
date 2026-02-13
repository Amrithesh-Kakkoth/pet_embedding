# Pet Face Embedding System — Technical Documentation

A lightweight, production-ready deep learning system for generating face embeddings of cats and dogs. Designed for pet photo applications where users need to automatically group, search, and match pet photos by identity.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Training Pipeline](#3-training-pipeline)
4. [Data Pipeline & Augmentations](#4-data-pipeline--augmentations)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Production Deployment](#6-production-deployment)
7. [Results & Benchmarks](#7-results--benchmarks)
8. [File Reference](#8-file-reference)

---

## 1. System Overview

### What It Does

The system takes a cropped, aligned photo of a pet face and produces a 128-dimensional embedding vector. Two photos of the same pet will have high cosine similarity (close to 1.0), while photos of different pets will have low similarity (close to 0.0). This enables:

- **Photo grouping**: Automatically cluster pet photos by identity (like Google Photos for faces)
- **Search**: "Find all photos of this pet" using a single reference photo
- **Verification**: "Are these two photos the same pet?" with a similarity threshold

### Key Design Constraints

| Constraint | Decision | Rationale |
|---|---|---|
| Mobile deployment | MobileNetV3-Small backbone | Must run on-device, not cloud |
| Model size | 308 KB (ONNX) | App store size limits, fast download |
| Inference speed | < 50ms on mobile | Real-time photo browsing |
| Accuracy | 79% recall@1, 0.99 ROC-AUC | Good enough for user-correctable grouping |

### System Components

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Aligned     │───>│  Embedder    │───>│  128-dim     │
│  Pet Face    │    │  (308 KB)    │    │  Embedding   │
│  224x224 RGB │    │  MobileNetV3 │    │  Unit Vector │
└─────────────┘    └──────────────┘    └──────────────┘
```

---

## 2. Architecture Design

### 2.1 Backbone: MobileNetV3-Small

**Choice**: `mobilenetv3_small_100` from the `timm` library (pre-trained on ImageNet).

**Why MobileNetV3-Small over alternatives**:

| Backbone | Params | ONNX Size | Why Not |
|---|---|---|---|
| **MobileNetV3-Small** | ~1.5M | **308 KB** | **Selected** — best size/accuracy tradeoff |
| MobileNetV3-Large | ~4.2M | ~1.2 MB | 4x larger for marginal accuracy gain |
| EfficientNet-B0 | ~5.3M | ~1.8 MB | Too large for mobile, overkill for pets |
| ResNet-18 | ~11.7M | ~45 MB | Far too large, designed for server-side |
| ShuffleNetV2 | ~1.4M | ~350 KB | Similar size but lower accuracy than MobileNetV3 |

MobileNetV3 uses **hardware-aware neural architecture search (NAS)** optimized for mobile latency. Key features:
- **Inverted residuals** with lightweight depthwise separable convolutions
- **Squeeze-and-excitation (SE) blocks** for channel attention
- **h-swish activation** — hardware-friendly approximation of swish
- **Pre-trained on ImageNet** — transfer learning gives strong low-level features (edges, textures, shapes) without training from scratch on millions of pet images

### 2.2 Embedding Head: GDConv (Global Depthwise Convolution)

**Choice**: MobileFaceNet-style GDConv head instead of standard Global Average Pooling (GAP) + MLP.

**Why GDConv over GAP + MLP**:

Standard approach (used in most classification models):
```
Feature Map (7x7x1024) → Global Average Pool → 1024-dim vector → MLP → 128-dim embedding
```
Problem: GAP collapses all spatial information. A feature in the top-left corner is treated identically to one in the bottom-right. For face/pet recognition, **where** a feature appears matters (e.g., eye position, ear shape, nose location).

Our approach (MobileFaceNet-style):
```
Feature Map (7x7x1024) → GDConv (7x7 depthwise) → 1024-dim → 1x1 Conv → 128-dim embedding
```

**GDConv** replaces GAP with a depthwise convolution whose kernel size equals the feature map dimensions (7x7). This means:
- Each spatial position gets a **learned weight** instead of equal averaging
- The model can learn that "eyes are usually in the upper portion" and weight those positions higher
- Adds negligible parameters (7 x 7 x 1024 = 50K weights) but preserves spatial structure

**Architecture detail**:
```python
class MobileFaceNetHead:
    GDConv(in_channels=1024, kernel_size=7)  # Spatial pooling with learned weights
    → Dropout(0.4)                            # Regularization
    → Conv1x1(1024 → 128)                    # Channel reduction to embedding dim
    → BatchNorm2d(128)                        # Normalize before L2 normalization
    → Flatten → L2 Normalize                 # Unit hypersphere embedding
```

### 2.3 Loss Function: ArcFace (Additive Angular Margin)

**Choice**: ArcFace with scale=32, margin=0.4, label smoothing=0.1.

**Why ArcFace over alternatives**:

| Loss Function | How It Works | Why Not |
|---|---|---|
| Triplet Loss | Pull positive pairs together, push negatives apart | Hard to mine good triplets, slow convergence, O(N^3) pairs |
| Contrastive Loss | Binary same/different pairs | Needs careful pair mining, less discriminative boundaries |
| Center Loss | Pull embeddings toward class center | Weak inter-class separation |
| Softmax | Standard classification | No explicit angular margin, poor generalization to unseen identities |
| **ArcFace** | Angular margin in cosine space | **Selected** — clean geometric interpretation, strong generalization |

**How ArcFace works**:

Standard softmax classification computes:
```
logit_i = W_i · x = ||W_i|| · ||x|| · cos(θ_i)
```

ArcFace normalizes both weights and embeddings to unit length, then adds an angular margin penalty to the target class:
```
logit_target = s · cos(θ_target + m)    ← harder: must be correct by margin m
logit_other  = s · cos(θ_other)          ← unchanged
```

Where `s=32` is a scaling factor and `m=0.4` is the angular margin in radians (~23 degrees).

This forces the model to learn embeddings where same-identity samples are clustered within a tight angular region, with at least 23 degrees of separation from other identities. The result is highly discriminative embeddings that generalize to identities never seen during training.

**Margin warmup**: The margin starts at 0 and linearly increases to 0.4 over the first 1/3 of training (13 epochs). This prevents early training instability when the model hasn't learned meaningful features yet.

### 2.4 Embedding Space

- **Dimensionality**: 128 (sweet spot between expressiveness and compactness)
- **Normalization**: L2-normalized to unit length (all embeddings lie on a 128-dimensional hypersphere)
- **Similarity metric**: Cosine similarity (equivalent to dot product for unit vectors)
- **Why 128 dims**: Standard in face recognition. 64 is too compressed for ~20K identities. 256+ adds storage cost with diminishing returns.

---

## 3. Training Pipeline

### 3.1 Training Configuration

```python
backbone:         "mobilenetv3_small_100"    # Pre-trained on ImageNet
embedding_dim:    128                         # Output embedding size
image_size:       224                         # Input resolution (increased from 160)
batch_size:       96                          # Fits in 20GB VRAM
epochs:           40                          # ~23 min (cat), ~27 min (dog)
learning_rate:    5e-4                        # With differential LR
weight_decay:     5e-4                        # AdamW regularization
dropout:          0.4                         # Strong dropout for small model
label_smoothing:  0.1                         # Soft targets for generalization
```

### 3.2 Training Strategy

**Differential learning rates**: The pre-trained backbone uses 10x lower learning rate (5e-5) than the randomly initialized embedding head and ArcFace classifier (5e-4). This preserves useful ImageNet features while allowing the head to adapt quickly.

```python
optimizer = AdamW([
    {"params": backbone_params, "lr": 5e-5},     # Pre-trained: careful updates
    {"params": head_params,     "lr": 5e-4},      # New layers: fast learning
])
```

**Learning rate schedule**:
1. **Linear warmup** (epochs 1-5): LR ramps from 0 to target, preventing early divergence
2. **Cosine annealing** (epochs 6-40): Smooth decay to 1e-6, avoids sharp LR drops

**ArcFace margin warmup** (epochs 1-13): Margin increases from 0 to 0.4 linearly. Early training focuses on basic classification; margin gradually enforces tighter angular separation.

**Gradient clipping**: Max norm = 1.0 to prevent exploding gradients from the angular margin loss.

### 3.3 Data Splits

Data is split **by identity** (not by image) to prevent data leakage:
- **Train**: 80% of identities (~15,965 cat / ~18,260 dog)
- **Validation**: 10% of identities — used for verification accuracy during training
- **Test**: 10% of identities — held out for final evaluation

This is critical: if images of the same pet appeared in both train and test, the model would appear to generalize but would fail on truly unseen pets.

### 3.4 Validation During Training

Every 3 epochs, the model is evaluated on 5,000 same/different pairs from the validation set. The best model by verification accuracy is saved as `best_model.pt`. This pair-based evaluation (rather than classification accuracy) better reflects the real-world use case.

---

## 4. Data Pipeline & Augmentations

### 4.1 Training Augmentations

Augmentations simulate real-world conditions that pet photos encounter. Each augmentation is applied with a probability `p` to maintain diversity.

```python
# Full augmentation pipeline (in order):
A.Resize(224, 224)

# --- Geometric transforms ---
A.Rotate(limit=25, p=0.5)              # Pets tilt their heads
A.HorizontalFlip(p=0.5)                # Left/right doesn't change identity
A.Affine(translate=10%, scale=0.85-1.15, p=0.3)  # Slight position/scale shifts

# --- Color transforms ---
A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5)
A.ToGray(p=0.05)                       # Forces learning shape over color

# --- Quality degradation ---
A.GaussianBlur(blur_limit=(3, 9), p=0.2)          # Phone camera blur, motion blur
A.GaussNoise(std_range=(0.01, 0.05), p=0.2)       # Sensor noise, low light
A.ImageCompression(quality_range=(70, 100), p=0.2) # JPEG artifacts from messaging apps
A.Downscale(scale_range=(0.4, 0.8), p=0.15)       # Low-res crops, zoom artifacts

# --- Occlusion ---
A.RandomShadow(p=0.1)                             # Partial lighting occlusion
A.CoarseDropout(holes=1-4, size=16-48px, p=0.3)   # Toys, hands, objects blocking face

# --- Normalization ---
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
ToTensorV2()
```

**Why these specific augmentations matter**:

| Augmentation | Real-World Scenario | Impact |
|---|---|---|
| GaussianBlur (3-9) | Motion blur, out-of-focus phone shots | Reduced blur similarity drop from 38% to 23% |
| CoarseDropout (16-48px) | Toys in mouth, hands petting, collar/leash | Reduced occlusion similarity drop from 53% to 44% |
| Downscale (0.4-0.8) | Cropped from group photo, zoomed in | Simulates resolution loss |
| RandomShadow | Indoor lighting, tree shadows outdoors | Partial brightness occlusion |
| ToGray (5%) | Forces model to learn shape, not just fur color | Prevents over-reliance on color for orange tabbies, etc. |

### 4.2 Validation/Test Transforms

No augmentation — only resize and normalize. This gives clean, reproducible evaluation metrics.

```python
A.Resize(224, 224)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ToTensorV2()
```

---

## 5. Evaluation Framework

The evaluation script (`evaluate_embeddings.py`) runs 8 benchmark suites covering different aspects of embedding quality.

### 5.1 Retrieval Benchmarks

**What**: Given a query image, find same-identity images in the test set.

| Metric | What It Measures | How |
|---|---|---|
| Recall@k (k=1,5,10) | % of queries where a correct match appears in top-k results | Higher = better |
| MRR (Mean Reciprocal Rank) | Average 1/rank of first correct result | Higher = better |
| NDCG@10 | Ranking quality weighted by position | Higher = better |

### 5.2 Clustering Quality

**What**: Run K-Means on embeddings and compare to true identity labels.

| Metric | What It Measures | Range |
|---|---|---|
| NMI (Normalized Mutual Information) | Agreement between clusters and labels | 0-1, higher = better |
| ARI (Adjusted Rand Index) | Pair-level cluster agreement, chance-adjusted | -1 to 1, higher = better |
| Purity | Fraction of samples matching majority label in their cluster | 0-1, higher = better |

### 5.3 Similarity Ranking (Verification)

**What**: Sample 50K image pairs, compute cosine similarity, evaluate binary same/different classification.

| Metric | What It Measures |
|---|---|
| ROC-AUC | Overall separability of same vs. different pairs |
| EER (Equal Error Rate) | Error rate where false positive rate = false negative rate |
| Spearman | Rank correlation between similarity and same/different labels |

### 5.4 Triplet Metrics

**What**: Sample 20K (anchor, positive, negative) triplets and check if the model ranks the positive closer.

| Metric | What It Measures |
|---|---|
| Violation Rate | % of triplets where negative is closer than positive |
| Average Margin | Mean (sim(a,p) - sim(a,n)), higher = better separation |

### 5.5 Overfitting Check

**What**: Compare recall@1 on training identities vs. test identities. A large gap indicates memorization rather than generalization.

### 5.6 Robustness Probes

**What**: Apply 5 degradations to test images and measure how much the embedding changes.

| Degradation | Severity | What It Simulates |
|---|---|---|
| Gaussian blur | radius=3.5 | Out-of-focus photos |
| Gaussian noise | std=0.05 | Low-light sensor noise |
| JPEG compression | quality=30 | Heavily compressed photos from messaging |
| Brightness shift | 1.5x | Over-exposed photos |
| Partial occlusion | 25% center block | Object blocking pet face |

Metrics: **similarity drop** (1 - cosine similarity between clean and degraded embeddings) and **recall@1 drop**.

### 5.7 Test-Time Augmentation (TTA)

When `--tta` flag is enabled, each image is embedded twice (original + horizontally flipped), the embeddings are averaged, and re-normalized to unit length. This provides a small but consistent accuracy boost (~0.3-0.4% recall@1) at the cost of 2x inference time.

```python
embs_original = model(images)
embs_flipped  = model(torch.flip(images, dims=[3]))  # Horizontal flip
embs_final    = F.normalize((embs_original + embs_flipped) / 2, p=2, dim=1)
```

### 5.8 Visualizations

The evaluation script generates:
- **Nearest-neighbor grid**: 10 query images with their top-5 matches (green border = correct, red = wrong)
- **Similarity distribution**: Histogram of positive vs. negative pair cosine similarities
- **ROC curve**: True positive rate vs. false positive rate
- **Robustness bar chart**: Similarity and recall drop per degradation type

---

## 6. Production Deployment

### 6.1 Exported Artifacts

| File | Size | Use |
|---|---|---|
| `pet_embedder.onnx` | 308 KB | Production inference (iOS CoreML, Android NNAPI, ONNX Runtime) |
| `pet_embedder.pt` | ~6.8 MB | PyTorch inference (server-side or research) |
| `best_model.pt` | ~44-47 MB | Full checkpoint (resume training, includes ArcFace head + optimizer) |

### 6.2 Inference Pipeline

```
1. Detect pet face (separate model, not included)
2. Crop and align face to 224x224 RGB
3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
4. Run through ONNX model → 128-dim float32 vector
5. L2-normalize the output (model does this internally, but verify)
6. Compare with stored embeddings using cosine similarity
```

### 6.3 Recommended Similarity Thresholds

Based on evaluation results:

| Use Case | Threshold | Behavior |
|---|---|---|
| High precision grouping | 0.65 | Few false matches, user merges missed groups |
| Balanced | 0.55 | Good balance of precision and recall |
| High recall search | 0.40 | Finds most matches, some false positives |

### 6.4 Separate Models for Cats and Dogs

Two separate models are trained — one for cats, one for dogs. This is intentional:

**Why not a single model?**
- Cat and dog faces have fundamentally different structures (whisker patterns, ear shapes, eye proportions)
- A single model would waste capacity learning to distinguish cats from dogs, rather than focusing on within-species identity
- Separate models allow each to specialize, yielding better accuracy with the same model size
- In production, species detection (cat vs dog) is a simple classification that routes to the correct embedder

---

## 7. Results & Benchmarks

### 7.1 Model Performance Summary

| Metric | Cat Model | Dog Model |
|---|---|---|
| Val Accuracy (verification) | 92.12% | 94.58% |
| Test Recall@1 | 73.57% | 78.59% |
| Test Recall@5 | 83.90% | 87.84% |
| Test Recall@10 | 87.22% | 90.61% |
| ROC-AUC | 0.9672 | 0.9874 |
| EER | 6.73% | 8.36% |
| NMI | 0.906 | 0.917 |
| Triplet Violation Rate | 3.54% | 1.98% |
| ONNX Model Size | 308 KB | 308 KB |

### 7.2 Robustness Results (Similarity Drop, lower = better)

| Degradation | Cat (prev) | Cat (new) | Dog (prev) | Dog (new) |
|---|---|---|---|---|
| Gaussian blur | 37.9% | **22.6%** | 36.5% | **20.2%** |
| Gaussian noise | 2.7% | 2.4% | 2.4% | 2.0% |
| JPEG compression | 4.5% | 4.9% | 4.0% | 3.7% |
| Brightness shift | 5.2% | 5.2% | 5.6% | 5.4% |
| Partial occlusion | 53.0% | **43.9%** | 48.8% | **39.7%** |

### 7.3 TTA Impact

| Metric | Cat (no TTA) | Cat (TTA) | Dog (no TTA) | Dog (TTA) |
|---|---|---|---|---|
| Recall@1 | 73.57% | 73.93% | 78.59% | 78.97% |
| ROC-AUC | 0.9672 | 0.9677 | 0.9874 | 0.9888 |
| Blur sim drop | 22.6% | 22.3% | 20.2% | 19.9% |

TTA provides a modest but consistent improvement. Recommended for batch processing (album organization) but optional for real-time use.

---

## 8. File Reference

### 8.1 Source Files

| File | Purpose | Key Classes/Functions |
|---|---|---|
| `config.py` | Training hyperparameters | `Config` dataclass |
| `model.py` | Neural network architecture | `PetFaceEmbedder`, `PetFaceModel`, `ArcFaceHead`, `GDConv`, `MobileFaceNetHead` |
| `dataset.py` | Data loading and augmentation | `PetFaceDataset`, `PairDataset`, `get_train_transforms()`, `load_dataset()` |
| `train.py` | Training loop | `train_epoch()`, `evaluate_verification()`, `export_embedder()` |
| `evaluate_embeddings.py` | Comprehensive evaluation | `compute_embeddings()`, `retrieval_metrics()`, `robustness_probes()`, 8 benchmark suites |

### 8.2 Data Layout

```
~/cat_aligned/                  # 19,957 identity folders
├── identity_001/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
├── identity_002/
└── ...

~/dog_aligned/                  # 22,826 identity folders
├── identity_001/
│   ├── img_001.png
│   └── ...
└── ...
```

### 8.3 Output Layout

```
outputs_cat/                    # Cat model outputs
├── best_model.pt               # Full training checkpoint (44 MB)
├── last_checkpoint.pt          # Last epoch checkpoint
├── pet_embedder.pt             # Embedder-only PyTorch weights
├── pet_embedder.onnx           # Production ONNX model (308 KB)
├── config.json                 # Training configuration
└── history.json                # Per-epoch training metrics

eval_cat_v2/                    # Evaluation results (no TTA)
├── eval_results.json           # All benchmark numbers
├── nearest_neighbors.png       # Visual NN grid
├── similarity_distribution.png # Positive/negative similarity histograms
├── roc_curve.png               # ROC curve
└── robustness.png              # Degradation robustness chart
```

### 8.4 Training & Evaluation Commands

```bash
# Activate environment
source ~/venv/bin/activate
cd ~/pet_embedding

# Train cat model
python train.py --data-dir ~/cat_aligned --output-dir outputs_cat

# Train dog model
python train.py --data-dir ~/dog_aligned --output-dir outputs_dog

# Evaluate without TTA
python evaluate_embeddings.py \
    --model-path outputs_cat/best_model.pt \
    --data-dir ~/cat_aligned \
    --output-dir eval_cat \
    --image-size 224

# Evaluate with TTA
python evaluate_embeddings.py \
    --model-path outputs_cat/best_model.pt \
    --data-dir ~/cat_aligned \
    --output-dir eval_cat_tta \
    --image-size 224 \
    --tta

# Resume training from checkpoint
python train.py --data-dir ~/cat_aligned --output-dir outputs_cat \
    --resume outputs_cat/last_checkpoint.pt
```

### 8.5 Environment

- **Hardware**: NVIDIA RTX 4000 Ada (20 GB VRAM)
- **Python**: 3.11
- **Key dependencies**: PyTorch 2.9.1, timm, albumentations, scikit-learn, ONNX
- **Training time**: ~23 min (cat, 40 epochs), ~27 min (dog, 40 epochs)
- **ONNX export**: opset_version=18 (required for PyTorch 2.9.1)
