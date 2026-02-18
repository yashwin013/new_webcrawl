"""
Text processing helpers for page-level operations.

Extracted from pipeline stages to work on individual pages in the worker system.
"""

import re
from typing import Optional, Tuple, List, Dict, Any

from app.crawling.models.document import (
    Page, PageContent, ContentSource, OCRAction, ImageInfo
)
from app.config import get_logger

logger = get_logger(__name__)


# ==================== Text Extraction ====================

def extract_page_text(page: Page, min_chars: int = 100, min_words: int = 20) -> PageContent:
    """
    Extract text content from a single page.
    
    Tries DOM text first, then HTML parsing, with cleaning and validation.
    
    Args:
        page: Page object with HTML/DOM content
        min_chars: Minimum characters for valid content
        min_words: Minimum words for valid content
        
    Returns:
        PageContent with extracted text
    """
    # Try DOM text first (fastest)
    if page.dom_text:
        cleaned = _clean_dom_text(page.dom_text)
        content = PageContent.from_text(cleaned, ContentSource.DOM)
        
        # Add images if available
        if page.scraped_images:
            content.images = [
                ImageInfo(
                    width=img.get("width", 0),
                    height=img.get("height", 0),
                    aspect_ratio=img.get("width", 0) / img.get("height", 1) if img.get("height", 0) > 0 else 0,
                    image_type="unknown",
                    area=img.get("width", 0) * img.get("height", 0)
                )
                for img in page.scraped_images
            ]
        return content
    
    # Try HTML content
    if page.html_content:
        text = _extract_from_html(page.html_content)
        cleaned = _clean_dom_text(text)
        return PageContent.from_text(cleaned, ContentSource.DOM)
    
    # No text available
    return PageContent(
        text="",
        word_count=0,
        char_count=0,
        source=ContentSource.DOM,
    )


def _clean_dom_text(raw_text: str) -> str:
    """
    Clean extracted DOM text.
    
    Removes:
    - Excessive whitespace
    - Navigation noise
    - Cookie banners
    """
    if not raw_text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", raw_text)
    
    # Remove common noise patterns
    noise_patterns = [
        r"Accept\s+cookies?",
        r"Cookie\s+policy",
        r"Privacy\s+policy",
        r"Terms\s+of\s+service",
        r"Skip\s+to\s+content",
        r"Toggle\s+navigation",
        r"Loading\.\.\.",
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove very short lines (likely buttons/links)
    lines = text.split("\n")
    cleaned_lines = [
        line.strip() for line in lines
        if len(line.strip()) > 20 or "." in line
    ]
    
    return "\n".join(cleaned_lines).strip()


def _extract_from_html(html: str) -> str:
    """Extract text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        return text
        
    except Exception as e:
        logger.error(f"HTML parsing failed: {e}")
        return ""


# ==================== OCR Decision ====================

def decide_ocr_for_page(
    page: Page,
    min_words: int = 100,
    scanned_max_words: int = 50,
    min_text_bearing_images: int = 3,
    min_text_bearing_ratio: float = 0.5,
) -> Tuple[OCRAction, str]:
    """
    Decide if OCR is needed for a single page.
    
    Returns:
        Tuple of (OCRAction, reason_string)
    """
    content = page.content
    if not content:
        return OCRAction.FULL_PAGE_OCR, "No content extracted"
    
    word_count = content.word_count
    text_bearing_images = len([
        img for img in content.images
        if _is_text_bearing(img, min_text_bearing_images)
    ])
    total_images = len(content.images)
    
    # Calculate ratios
    text_bearing_ratio = (
        text_bearing_images / total_images if total_images > 0 else 0.0
    )
    
    # Decision logic
    
    # Case 1: Very little text - likely scanned document
    if word_count < scanned_max_words:
        return (
            OCRAction.FULL_PAGE_OCR,
            f"Low word count ({word_count} < {scanned_max_words})"
        )
    
    # Case 2: Good text but many text-bearing images
    if word_count >= min_words:
        if (text_bearing_images >= min_text_bearing_images or
            text_bearing_ratio >= min_text_bearing_ratio):
            return (
                OCRAction.OCR_IMAGES_ONLY,
                f"Good text ({word_count} words) + {text_bearing_images} text images"
            )
        else:
            return (
                OCRAction.SKIP_OCR,
                f"Sufficient text ({word_count} words), no significant images"
            )
    
    # Case 3: Moderate text with images - check if images dominate
    if text_bearing_ratio > 0.7:
        return (
            OCRAction.FULL_PAGE_OCR,
            f"Image-heavy page ({text_bearing_ratio:.0%} text-bearing)"
        )
    
    # Case 4: Moderate text, some images
    if text_bearing_images > 0:
        return (
            OCRAction.OCR_IMAGES_ONLY,
            f"Moderate text ({word_count} words) + some images"
        )
    
    # Default: Skip OCR
    return (
        OCRAction.SKIP_OCR,
        f"Moderate text ({word_count} words), no text-bearing images"
    )


def _is_text_bearing(img: ImageInfo, min_text_bearing: int = 3) -> bool:
    """Check if image likely contains text."""
    # Large images are more likely to contain text
    if img.area > 50000:
        return True
    
    # Wide/tall images (tables, charts)
    if img.aspect_ratio > 2.0 or img.aspect_ratio < 0.5:
        return True
    
    return False


# ==================== Chunking ====================
# NOTE: Sentence-based chunking has been REMOVED in favor of Docling's HybridChunker.
# All chunking now happens in pdf_processor.py using structure-aware HybridChunker.
# This provides better quality chunks with document structure preservation.
#
# If you need chunking functionality:
# - For PDF/web pages: Use app.docling.processor.AsyncDocumentProcessor.create_chunks_async()
# - For plain text: Consider implementing a wrapper around HybridChunker
#
# Legacy imports (OCR worker/scripts) need to be updated to use the new approach.
