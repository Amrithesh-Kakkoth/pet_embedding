# Pet Face Embedding

Embedding models for pet (cat/dog) face recognition. Trains compact neural networks with ArcFace metric learning to produce 128-d embeddings suitable for identity matching and clustering.

## Models

Two backbone architectures, each trained separately on cats and dogs:

| Backbone | Params | Cat Val Acc | Dog Val Acc |
|---|---|---|---|
| MobileNetV3-Small + GDConv | 1.7M | 92.2% | 94.7% |
| MobileFaceNet | 1.1M | — | — |

## Project Structure

```
src/            Core modules (model, dataset, config)
training/       Training scripts (MobileNetV3, MobileFaceNet)
evaluation/     Embedding evaluation & model comparison
pipeline/       Face clustering (HDBSCAN) for Battersea dataset
deploy/         ONNX export & inference pipeline
scripts/        Shell scripts for training runs
docs/           Detailed documentation & model comparison
tests/          Setup verification tests
```

## Quick Start

```bash
pip install -r requirements.txt

# Train cat model
python training/train.py --data-dir ~/cat_aligned --output-dir outputs_cat

# Train dog model
python training/train.py --data-dir ~/dog_aligned --output-dir outputs_dog

# Evaluate
python evaluation/evaluate_embeddings.py --checkpoint outputs_cat/pet_embedder.pt --data-dir ~/cat_aligned

# Export to ONNX
python deploy/export_mobile.py --checkpoint outputs_cat/pet_embedder.pt
```

## Data

Expects pre-aligned face crops organized as `data_dir/<identity_id>/image.jpg`. Training uses an 80/10/10 train/val/test split.

## Requirements

PyTorch 2.0+, timm, albumentations, onnxruntime. See `requirements.txt`.
