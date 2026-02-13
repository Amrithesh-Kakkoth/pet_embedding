#!/bin/bash
# Train separate embedding models for cats and dogs.
# Uses the proven hyperparameters from the best dog-only run (outputs_224).

set -e

# Activate virtual environment
source ~/venv/bin/activate

cd ~/pet_embedding

echo "============================================"
echo "  Training Cat Embedding Model"
echo "============================================"
python training/train.py \
    --data-dir ~/cat_aligned \
    --output-dir outputs_cat \
    --backbone mobilenetv3_small_100 \
    --head-type gdconv \
    --epochs 40 \
    --batch-size 96 \
    --lr 5e-4

echo ""
echo "============================================"
echo "  Training Dog Embedding Model"
echo "============================================"
python training/train.py \
    --data-dir ~/dog_aligned \
    --output-dir outputs_dog \
    --backbone mobilenetv3_small_100 \
    --head-type gdconv \
    --epochs 40 \
    --batch-size 96 \
    --lr 5e-4

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo "Cat model: outputs_cat/"
echo "Dog model: outputs_dog/"
