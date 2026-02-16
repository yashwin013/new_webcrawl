#!/bin/bash
# Process OCR Backlog with Optimized GPU Memory Settings
# Sets PyTorch memory allocation to reduce fragmentation

echo "======================================================================"
echo "OCR Backlog Processing - GPU Memory Optimized"
echo "======================================================================"
echo ""
echo "Setting PyTorch memory configuration..."
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/process_ocr_backlog.py "$@"
