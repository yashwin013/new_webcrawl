"""
Worker helper modules for page-level processing.

NOTE: chunk_page_text has been removed. Use HybridChunker instead:
    from app.docling.processor import AsyncDocumentProcessor
    processor = AsyncDocumentProcessor(...)
    await processor.initialize_async()
    doc = await processor.convert_document_async(pdf_path)
    chunks = await processor.create_chunks_async()
"""

from app.orchestrator.workers.helpers.text_processor import (
    extract_page_text,
    decide_ocr_for_page,
)
from app.orchestrator.workers.helpers.ocr_processor import (
    process_page_ocr,
)

__all__ = [
    "extract_page_text",
    "decide_ocr_for_page",
    "process_page_ocr",
]
