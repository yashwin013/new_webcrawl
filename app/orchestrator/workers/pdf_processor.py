"""
PDF Processor Worker

Processes PDF files using Docling (GPU) to extract text.
GPU-intensive worker - should have minimal concurrency (1-2 workers max).
"""

import aiofiles
import asyncio
import time
from typing import Optional
from pathlib import Path

from app.config import get_logger, OCR_MIN_WORD_COUNT_SUFFICIENT
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import PdfTask, StorageTask
from app.orchestrator.workers.helpers.text_processor import chunk_page_text
from app.docling.processor import AsyncDocumentProcessor
from app.crawling.models.document import Page, PageContent, ContentSource

logger = get_logger(__name__)


class PdfProcessorWorker(BaseWorker):
    """
    PDF processor worker - handles GPU-based PDF text extraction using Docling.
    
    Responsibilities (GPU):
    - Pull PdfTask from pdf_queue
    - Extract text from PDF using Docling (GPU)
    - Chunk the extracted text
    - Push chunks to storage_queue
    - If extraction fails or minimal text: mark for OCR backlog
    
    IMPORTANT: Only 1-2 workers should run to prevent GPU overload!
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        min_chunk_words: int = 100,
        max_chunk_words: int = 512,
        overlap_words: int = 50,
        timeout_seconds: int = 120,
    ):
        super().__init__(worker_id, queue_manager)
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.timeout_seconds = timeout_seconds
        
        # Docling processor (lazy initialization)
        self._docling_processor = None
        
        # Statistics
        self.total_processing_time = 0.0
        self.pdfs_processed = 0
        self.pdfs_failed = 0
    
    @property
    def worker_type(self) -> str:
        return "pdf_processor"
    
    async def startup(self):
        """Initialize PDF processor resources (loads GPU models)."""
        await super().startup()
        logger.info(f"[{self.worker_id}] PDF processor worker initialized (GPU)")
    
    async def get_next_task(self) -> Optional[PdfTask]:
        """Get next PDF task from queue."""
        return await self.queue_manager.get_pdf_task(timeout=2.0)
    
    async def process_task(self, task: PdfTask) -> bool:
        """
        Extract text from PDF using Docling and prepare for storage.
        
        Args:
            task: PdfTask with PDF to process
            
        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)
        
        page = task.page
        pdf_path = page.pdf_path
        
        if not pdf_path or not pdf_path.exists():
            logger.error(f"[{self.worker_id}] PDF not found: {pdf_path}")
            task.mark_failed("PDF file not found")
            return False
        
        logger.info(f"[{self.worker_id}] Processing PDF: {page.url}")
        
        start_time = time.time()
        
        try:
            # Extract text from PDF with timeout
            text = await asyncio.wait_for(
                self._extract_pdf_text(pdf_path),
                timeout=self.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.pdfs_processed += 1
            
            word_count = len(text.split()) if text else 0
            if text and word_count > OCR_MIN_WORD_COUNT_SUFFICIENT:  # Has meaningful text
                # Update page content
                page.content = PageContent.from_text(text, ContentSource.TEXT_LAYER)
                
                logger.info(
                    f"[{self.worker_id}] Extracted {len(text)} chars ({word_count} words) from PDF "
                    f"in {processing_time:.2f}s: {page.url}"
                )
                
                # Optional: Save as markdown file
                # Extract file_id from URL or use PDF filename
                file_id = page.url_hash if hasattr(page, 'url_hash') else pdf_path.stem
                markdown_path = await self._save_markdown(pdf_path, file_id)
                if markdown_path:
                    logger.info(f"[{self.worker_id}] Saved markdown: {markdown_path}")
                
                # Chunk the text
                chunks = chunk_page_text(
                    page,
                    min_words=self.min_chunk_words,
                    max_words=self.max_chunk_words,
                    overlap_words=self.overlap_words,
                )
                
                if chunks:
                    # Create storage task
                    storage_task = StorageTask(
                        task_id=f"{task.task_id}_storage",
                        website_url=task.website_url,
                        crawl_session_id=task.crawl_session_id,
                        chunks=chunks,
                        document_metadata={
                            "url": page.url,
                            "depth": page.depth,
                            "pdf_processed": True,
                            "processing_time_seconds": processing_time,
                        },
                        priority=task.priority,
                    )
                    
                    await self.queue_manager.put_storage_task(storage_task)
                    
                    logger.info(
                        f"[{self.worker_id}] Created {len(chunks)} chunks "
                        f"from PDF: {page.url}"
                    )
                else:
                    logger.warning(f"[{self.worker_id}] No chunks created from PDF: {page.url}")
                
                task.mark_completed()
                return True
                
            else:
                # Minimal or no text - likely scanned PDF
                word_count = len(text.split()) if text else 0
                logger.warning(
                    f"[{self.worker_id}] Minimal text extracted from PDF ({word_count} words, threshold: {OCR_MIN_WORD_COUNT_SUFFICIENT}). "
                    f"Marking for OCR: {page.url}"
                )
                
                # Save to OCR backlog
                await self._save_to_ocr_backlog(page, task.website_url)
                
                task.mark_completed()
                return True
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(
                f"[{self.worker_id}] PDF processing timeout ({self.timeout_seconds}s) "
                f"for {page.url} - marking for OCR"
            )
            
            # Timeout - save to OCR backlog
            await self._save_to_ocr_backlog(page, task.website_url)
            
            task.mark_completed()
            return True  # Don't fail the task, just skip Docling
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.pdfs_failed += 1
            
            logger.error(
                f"[{self.worker_id}] PDF processing failed for {page.url} "
                f"after {processing_time:.2f}s: {e}",
                exc_info=True
            )
            
            # Try to save to OCR backlog as fallback
            try:
                await self._save_to_ocr_backlog(page, task.website_url)
                logger.info(f"[{self.worker_id}] Saved to OCR backlog as fallback: {page.url}")
                task.mark_completed()
                return True
            except:
                pass
            
            task.mark_failed(str(e))
            return False
    
    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using Docling.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        # Lazy initialization of Docling processor with GPU
        if self._docling_processor is None:
            self._docling_processor = AsyncDocumentProcessor(
                use_gpu=True,  # Enable GPU for Phase 1 PDF processing
                executor=None,  # Will use executor from manager
                skip_ocr=True,  # CRITICAL: Disable OCR in Phase 1 to prevent memory exhaustion
            )
        
        # Use async conversion method
        result = await self._docling_processor.convert_document_async(
            str(pdf_path)
        )
        
        # Extract text from the DoclingDocument
        if result and hasattr(result, 'export_to_markdown'):
            return result.export_to_markdown()
        
        return ""
    
    async def _save_markdown(self, pdf_path: Path, file_id: str) -> Optional[Path]:
        """
        Extract and save PDF as markdown file.
        
        Args:
            pdf_path: Path to PDF file
            file_id: Unique identifier for the file
            
        Returns:
            Path to saved markdown file, or None if failed
        """
        try:
            # Lazy initialization of Docling processor with GPU
            if self._docling_processor is None:
                self._docling_processor = AsyncDocumentProcessor(
                    use_gpu=True,
                    executor=None,
                    skip_ocr=True,
                )
            
            # Convert document
            result = await self._docling_processor.convert_document_async(
                str(pdf_path)
            )
            
            # Export to markdown
            if result and hasattr(result, 'export_to_markdown'):
                markdown_content = result.export_to_markdown()
                
                # Create markdown output directory
                markdown_dir = Path("outputs/markdown")
                markdown_dir.mkdir(parents=True, exist_ok=True)
                
                # Save markdown file
                markdown_path = markdown_dir / f"{file_id}.md"
                import aiofiles
                async with aiofiles.open(markdown_path, 'w', encoding='utf-8') as f:
                    await f.write(markdown_content)
                
                logger.info(f"[{self.worker_id}] Saved markdown: {markdown_path}")
                return markdown_path
            
            return None
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to save markdown: {e}", exc_info=True)
            return None
    
    async def _save_to_ocr_backlog(self, page: Page, website_url: str):
        """Save page to OCR backlog for Phase 2 processing."""
        import json
        from datetime import datetime
        
        backlog_dir = Path("outputs/ocr_backlog")
        backlog_dir.mkdir(parents=True, exist_ok=True)
        
        backlog_file = backlog_dir / "pending_ocr.jsonl"
        
        entry = {
            "url": page.url,
            "pdf_path": str(page.pdf_path) if page.pdf_path else None,
            "ocr_action": "full_page_ocr",
            "website_url": website_url,
            "depth": page.depth,
            "timestamp": datetime.utcnow().isoformat(),
            "url_hash": page.url_hash if hasattr(page, 'url_hash') else None,
            "reason": "docling_extraction_failed",
        }
        
        # Append to JSONL file
        with backlog_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.info(f"[{self.worker_id}] Saved to OCR backlog: {page.url}")
    
    @property
    def stats(self) -> dict:
        """Get PDF processor worker statistics."""
        base_stats = super().stats
        
        avg_processing_time = (
            self.total_processing_time / self.pdfs_processed
            if self.pdfs_processed > 0 else 0
        )
        
        base_stats.update({
            "pdfs_processed": self.pdfs_processed,
            "pdfs_failed": self.pdfs_failed,
            "total_processing_time_seconds": self.total_processing_time,
            "avg_processing_time_seconds": avg_processing_time,
        })
        
        return base_stats
