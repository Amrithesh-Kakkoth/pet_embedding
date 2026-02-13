#!/bin/bash
# Quick start training script

set -e

# Default values
DATA_DIR="${DATA_DIR:-$HOME/Downloads/dog}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo "=== Pet Face Embedding Training ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Count identities
NUM_IDS=$(find "$DATA_DIR" -maxdepth 1 -type d | wc -l)
echo "Found ~$NUM_IDS dog identities"
echo ""

# Run training
python training/train.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "=== Training Complete ==="
echo "Models saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Export for mobile: python deploy/export_mobile.py --checkpoint $OUTPUT_DIR/pet_embedder.pt"
echo "  2. Use in inference: see deploy/inference.py"
