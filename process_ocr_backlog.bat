@echo off
REM Process OCR Backlog with Optimized GPU Memory Settings
REM Sets PyTorch memory allocation to reduce fragmentation

echo ======================================================================
echo OCR Backlog Processing - GPU Memory Optimized
echo ======================================================================
echo.
echo Setting PyTorch memory configuration...
echo   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo.

SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts\process_ocr_backlog.py %*
