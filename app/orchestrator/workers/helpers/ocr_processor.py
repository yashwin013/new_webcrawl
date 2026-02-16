"""
OCR processing helpers for page-level operations.

Extracted from OCRProcessorStage to work on individual pages.
"""

import asyncio
import io
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from app.crawling.models.document import Page, PageContent, OCRAction, ContentSource
from app.services.gpu_manager import get_surya_predictors
from app.config import get_logger, OCR_MAX_BBOXES_PER_PAGE

logger = get_logger(__name__)


async def process_page_ocr(
    page: Page,
    ocr_action: OCRAction = OCRAction.FULL_PAGE_OCR,
) -> Optional[str]:
    """
    Perform OCR on a single page.
    
    Args:
        page: Page object with PDF path
        ocr_action: Type of OCR to perform
        
    Returns:
        Extracted text from OCR, or None if failed
    """
    if not page.pdf_path or not page.pdf_path.exists():
        logger.warning(f"No PDF for OCR: {page.url}")
        return None
    
    start_time = time.time()
    
    try:
        if ocr_action == OCRAction.FULL_PAGE_OCR:
            text = await _run_full_page_ocr(page.pdf_path)
        else:
            text = await _run_images_only_ocr(page.pdf_path)
        
        processing_time = time.time() - start_time
        logger.info(f"OCR completed in {processing_time:.2f}s: {page.url}")
        
        return text
        
    except Exception as e:
        error_str = str(e)
        is_cuda_error = any(keyword in error_str.lower() for keyword in 
                           ['cuda', 'device-side assert', 'index out of bounds', 'gpu'])
        
        if is_cuda_error:
            logger.error(f"CUDA OCR error for {page.url}: {e}")
            # Don't propagate CUDA errors as they corrupt future operations
            # The models have already been cleared by _run_full_page_ocr error handler
            return None
        else:
            logger.error(f"OCR failed for {page.url}: {e}")
            return None


async def _run_full_page_ocr(pdf_path: Path) -> str:
    """Run full-page OCR on PDF using Surya with batch processing for large PDFs."""
    try:
        import fitz  # PyMuPDF
        import gc
        import torch
        
        # Use centralized GPU manager (singleton)
        try:
            det_predictor, rec_predictor = get_surya_predictors()
        except Exception as e:
            logger.error(f"Failed to load Surya OCR models: {e}")
            logger.warning("OCR unavailable - skipping OCR processing")
            return ""  # Return empty string to continue pipeline
        
        # Check for CUDA corruption and reset if needed
        if torch.cuda.is_available():
            try:
                # Test CUDA state with a simple operation
                test_tensor = torch.zeros(1, device='cuda')
                _ = test_tensor + 1
                del test_tensor
            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e):
                    logger.warning(f"Detected corrupted CUDA state: {e}")
                    logger.info("Attempting to recover by reinitializing OCR models...")
                    from app.services.gpu_manager import clear_models
                    clear_models()
                    torch.cuda.empty_cache()
                    # Try to get fresh models
                    det_predictor, rec_predictor = get_surya_predictors()
                    logger.info("✓ OCR models reinitialized successfully")
        
        # Extract images from PDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        # Process in batches to avoid GPU OOM on large PDFs
        BATCH_SIZE = 10  # Process 10 pages at a time
        all_page_texts = {}
        skipped_pages = {}
        
        logger.info(f"Processing {total_pages} pages in batches of {BATCH_SIZE}...")
        
        for batch_start in range(0, total_pages, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_pages)
            batch_size = batch_end - batch_start
            
            logger.debug(f"Processing batch: pages {batch_start+1}-{batch_end}/{total_pages}")
            
            # Extract images for this batch only
            images = []
            for page_num in range(batch_start, batch_end):
                page = doc[page_num]
                # Lower DPI from 150 to 96 for speed/noise reduction
                pix = page.get_pixmap(dpi=96)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            # Preprocess images (binarization to reduce noise)
            processed_images = []
            for img in images:
                # Convert to grayscale
                img = img.convert('L') 
                # Binarize - threshold 200 to reduce noise
                img = img.point(lambda p: 255 if p > 200 else 0)
                processed_images.append(img.convert('RGB'))
            
            # Run Detection first
            loop = asyncio.get_event_loop()
            det_predictions = await loop.run_in_executor(
                None,
                det_predictor,
                processed_images
            )
            
            # Check for complexity (too many bboxes = noise/hallucination)
            per_page_counts = [len(pred.bboxes) for pred in det_predictions]
            total_boxes = sum(per_page_counts)
            logger.debug(f"Detected {total_boxes} text regions in batch")
            
            final_images_to_recognize = []
            final_indices = []
            batch_placeholders = {}
            
            for i, pred in enumerate(det_predictions):
                page_idx = batch_start + i
                # Skip pages with too many bboxes (likely noise)
                if len(pred.bboxes) > OCR_MAX_BBOXES_PER_PAGE:
                    skipped_pages[page_idx] = len(pred.bboxes)
                    logger.warning(
                        f"Page {page_idx+1} has {len(pred.bboxes)} text regions "
                        f"(limit: {OCR_MAX_BBOXES_PER_PAGE}). Skipping to prevent hanging."
                    )
                    batch_placeholders[i] = f"[SKIPPED_COMPLEX_PAGE: {len(pred.bboxes)} regions]"
                else:
                    final_images_to_recognize.append(processed_images[i])
                    final_indices.append(i)
            
            if not final_images_to_recognize:
                # All pages in batch skipped
                for i in range(batch_size):
                    page_idx = batch_start + i
                    all_page_texts[page_idx] = batch_placeholders.get(i, "")
                continue
            
            # Prepare polygons for recognition
            polygons_to_recognize = []
            for idx in final_indices:
                page_polygons = []
                for bbox in det_predictions[idx].bboxes:
                    # Extract polygon coordinates
                    if hasattr(bbox, 'polygon'):
                        page_polygons.append(bbox.polygon)
                    elif hasattr(bbox, 'bbox'):
                        # bbox is [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox.bbox
                        # Convert to polygon format
                        page_polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    else:
                        # Try to convert directly
                        try:
                            if isinstance(bbox, list):
                                page_polygons.append(bbox)
                            else:
                                page_polygons.append(bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))
                        except:
                            logger.warning(f"Could not convert bbox: {type(bbox)}")
                            continue
                polygons_to_recognize.append(page_polygons)
            
            # Run Recognition on safe pages
            rec_predictions = await loop.run_in_executor(
                None,
                lambda: rec_predictor(final_images_to_recognize, polygons=polygons_to_recognize)
            )
            
            # Extract text from predictions
            for rec_pred, batch_idx in zip(rec_predictions, final_indices):
                page_idx = batch_start + batch_idx
                page_text_lines = []
                if hasattr(rec_pred, 'text_lines'):
                    for line in rec_pred.text_lines:
                        if hasattr(line, 'text'):
                            page_text_lines.append(line.text)
                        else:
                            page_text_lines.append(str(line))
                
                all_page_texts[page_idx] = "\n".join(page_text_lines)
            
            # Add skipped pages from this batch
            for batch_idx, placeholder in batch_placeholders.items():
                page_idx = batch_start + batch_idx
                all_page_texts[page_idx] = placeholder
            
            # Clear GPU cache between batches to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.debug(f"✓ Completed batch {batch_start+1}-{batch_end}. GPU memory cleared.")
        
        doc.close()
        
        # Combine all pages in order
        sorted_text = [all_page_texts.get(i, "") for i in range(total_pages)]
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return "\n\n".join(sorted_text)
        
    except Exception as e:
        logger.error(f"Full page OCR failed: {e}")
        
        # Check if this is a CUDA error - if so, clear corrupted models
        error_str = str(e)
        is_cuda_error = any(keyword in error_str.lower() for keyword in 
                           ['cuda', 'device-side assert', 'index out of bounds', 'gpu'])
        
        if is_cuda_error:
            logger.warning("CUDA error detected - clearing corrupted OCR models for recovery")
            try:
                from app.services.gpu_manager import clear_models
                clear_models()
                logger.info("✓ Corrupted models cleared - next OCR will use fresh models")
            except Exception as clear_error:
                logger.error(f"Failed to clear models: {clear_error}")
        
        # Clean up GPU memory on error
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass
        raise


async def _run_images_only_ocr(pdf_path: Path) -> str:
    """
    Run OCR only on extracted images (not full page).
    
    This is a simplified version for pages with good text but some images.
    """
    try:
        import fitz
        
        try:
            det_predictor, rec_predictor = get_surya_predictors()
        except Exception as e:
            logger.error(f"Failed to load Surya OCR models: {e}")
            logger.warning("OCR unavailable - skipping image OCR")
            return ""  # Return empty string to continue pipeline
        
        # Extract images from PDF
        doc = fitz.open(str(pdf_path))
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(image_bytes))
                
                # Only process larger images (likely contain text)
                if img.width > 200 and img.height > 100:
                    images.append(img)
        
        doc.close()
        
        if not images:
            return ""
        
        logger.debug(f"Running OCR on {len(images)} extracted images...")
        
        # Run detection and recognition
        loop = asyncio.get_event_loop()
        det_predictions = await loop.run_in_executor(None, det_predictor, images)
        
        # Prepare polygons
        polygons = []
        for pred in det_predictions:
            page_polygons = []
            for bbox in pred.bboxes:
                if hasattr(bbox, 'polygon'):
                    page_polygons.append(bbox.polygon)
                elif hasattr(bbox, 'bbox'):
                    x1, y1, x2, y2 = bbox.bbox
                    page_polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            polygons.append(page_polygons)
        
        rec_predictions = await loop.run_in_executor(
            None,
            lambda: rec_predictor(images, polygons=polygons)
        )
        
        # Extract text
        all_text = []
        for rec_pred in rec_predictions:
            if hasattr(rec_pred, 'text_lines'):
                image_text = "\n".join([
                    line.text if hasattr(line, 'text') else str(line)
                    for line in rec_pred.text_lines
                ])
                all_text.append(image_text)
        
        return "\n\n".join(all_text)
        
    except Exception as e:
        logger.error(f"Images-only OCR failed: {e}")
        
        # Check if this is a CUDA error - if so, clear corrupted models
        error_str = str(e)
        is_cuda_error = any(keyword in error_str.lower() for keyword in 
                           ['cuda', 'device-side assert', 'index out of bounds', 'gpu'])
        
        if is_cuda_error:
            logger.warning("CUDA error detected - clearing corrupted OCR models for recovery")
            try:
                from app.services.gpu_manager import clear_models
                clear_models()
                logger.info("✓ Corrupted models cleared - next OCR will use fresh models")
            except Exception as clear_error:
                logger.error(f"Failed to clear models: {clear_error}")
        
        # Clean up GPU memory on error
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except:
            pass
        raise
