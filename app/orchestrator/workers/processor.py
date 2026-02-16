"""
Processor Worker

Processes crawled pages: text extraction, OCR decision, and chunking.
CPU-intensive worker.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import ProcessTask, OcrTask, StorageTask, PdfTask
from app.orchestrator.workers.helpers.text_processor import (
    extract_page_text,
    decide_ocr_for_page,
    chunk_page_text,
)
from app.crawling.models.document import OCRAction, Page

logger = get_logger(__name__)


class ProcessorWorker(BaseWorker):
    """
    Processor worker - handles page text extraction and chunking.
    
    Responsibilities (CPU):
    - Pull ProcessTask from processing_queue
    - Extract text from HTML/DOM
    - Decide if OCR is needed
    - If OCR needed: push to ocr_queue
    - If no OCR: chunk text and push to storage_queue
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        min_chunk_words: int = 100,
        max_chunk_words: int = 512,
        overlap_words: int = 50,
        enable_ocr: bool = True,  # If False, skip OCR queue and chunk everything
    ):
        super().__init__(worker_id, queue_manager)
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.enable_ocr = enable_ocr
    
    @property
    def worker_type(self) -> str:
        return "processor"
    
    async def get_next_task(self) -> Optional[ProcessTask]:
        """Get next process task from queue."""
        return await self.queue_manager.get_process_task(timeout=1.0)
    
    async def process_task(self, task: ProcessTask) -> bool:
        """
        Process a page: extract text, decide OCR, chunk.
        
        Args:
            task: ProcessTask with page to process
            
        Returns:
            True if successful
        """
        # Defensive check: ensure task is correct type
        if not isinstance(task, ProcessTask):
            logger.error(
                f"[{self.worker_id}] Received wrong task type: {type(task).__name__}. "
                f"Expected ProcessTask. Skipping task."
            )
            return False
        
        task.mark_started(self.worker_id)
        
        page = task.page
        logger.debug(f"[{self.worker_id}] Processing {page.url}")
        
        try:
            # Check if this is a PDF page
            if page.pdf_path and page.pdf_path.exists():
                # Check PDF size - skip processing for large files
                pdf_size_mb = page.pdf_path.stat().st_size / (1024 * 1024)
                
                from app.config import PDF_MAX_PROCESSING_SIZE_MB
                
                if PDF_MAX_PROCESSING_SIZE_MB > 0 and pdf_size_mb > PDF_MAX_PROCESSING_SIZE_MB:
                    # Large PDF - save to separate folder and skip processing
                    logger.info(
                        f"[{self.worker_id}] Large PDF detected ({pdf_size_mb:.2f} MB > {PDF_MAX_PROCESSING_SIZE_MB} MB): {page.url}. "
                        "Saving to storage only, skipping processing."
                    )
                    
                    await self._save_large_pdf_to_storage(page, task.website_url, pdf_size_mb)
                    task.mark_completed(needs_ocr=False, chunks=0)
                    
                    logger.info(f"[{self.worker_id}] Large PDF saved to storage (not processed): {page.url}")
                    return True
                
                # Normal size PDF - route to PDF processor
                logger.debug(f"[{self.worker_id}] Page has PDF ({pdf_size_mb:.2f} MB), routing to PDF processor: {page.url}")
                
                pdf_task = PdfTask(
                    task_id=f"{task.task_id}_pdf",
                    page=page,
                    website_url=task.website_url,
                    crawl_session_id=task.crawl_session_id,
                    priority=task.priority,
                )
                
                await self.queue_manager.put_pdf_task(pdf_task)
                task.mark_completed(needs_ocr=False, chunks=0)
                
                logger.debug(f"[{self.worker_id}] Routed to PDF queue: {page.url}")
                return True
            
            # Step 1: Extract text from page (CPU)
            content = extract_page_text(page)
            page.content = content
            task.text_extracted = True
            
            logger.debug(
                f"[{self.worker_id}] Extracted {content.word_count} words from {page.url}"
            )
            
            # Step 2: Decide if OCR is needed (CPU)
            ocr_action, ocr_reason = decide_ocr_for_page(
                page,
                min_words=100,
                scanned_max_words=50,
            )
            page.ocr_action = ocr_action
            page.ocr_reason = ocr_reason
            
            logger.debug(
                f"[{self.worker_id}] OCR decision for {page.url}: {ocr_action.value} - {ocr_reason}"
            )
            
            # Step 3: Route based on OCR decision
            if self.enable_ocr and ocr_action in (OCRAction.FULL_PAGE_OCR, OCRAction.OCR_IMAGES_ONLY):
                # Needs OCR - push to OCR queue (GPU worker will handle)
                ocr_task = OcrTask(
                    task_id=f"{task.task_id}_ocr",
                    page=page,
                    website_url=task.website_url,
                    crawl_session_id=task.crawl_session_id,
                    ocr_action=ocr_action,
                    priority=task.priority,
                )
                
                await self.queue_manager.put_ocr_task(ocr_task)
                task.mark_completed(needs_ocr=True)
                
                logger.debug(f"[{self.worker_id}] Routed to OCR queue: {page.url}")
                
            elif not self.enable_ocr and ocr_action in (OCRAction.FULL_PAGE_OCR, OCRAction.OCR_IMAGES_ONLY):
                # OCR disabled but page needs it - save to pending OCR list
                await self._save_to_ocr_backlog(page, ocr_action, task.website_url)
                # Still chunk with existing content
                chunks = chunk_page_text(
                    page,
                    min_words=self.min_chunk_words,
                    max_words=self.max_chunk_words,
                    overlap_words=self.overlap_words,
                )
                
                if chunks:
                    storage_task = StorageTask(
                        task_id=f"{task.task_id}_storage",
                        website_url=task.website_url,
                        crawl_session_id=task.crawl_session_id,
                        chunks=chunks,
                        document_metadata={
                            "url": page.url,
                            "depth": page.depth,
                            "ocr_action": ocr_action.value,
                            "ocr_pending": True,  # Mark for later OCR
                        },
                        priority=task.priority,
                    )
                    
                    await self.queue_manager.put_storage_task(storage_task)
                    task.mark_completed(needs_ocr=False, chunks=len(chunks))
                    
                    logger.debug(
                        f"[{self.worker_id}] Created {len(chunks)} chunks (OCR pending), "
                        f"routed to storage: {page.url}"
                    )
                else:
                    task.mark_completed(needs_ocr=False, chunks=0)
                    logger.debug(f"[{self.worker_id}] No chunks created for {page.url}")
                
            else:
                # No OCR needed - chunk and push to storage
                # No OCR needed - chunk and push to storage
                chunks = chunk_page_text(
                    page,
                    min_words=self.min_chunk_words,
                    max_words=self.max_chunk_words,
                    overlap_words=self.overlap_words,
                )
                
                if chunks:
                    storage_task = StorageTask(
                        task_id=f"{task.task_id}_storage",
                        website_url=task.website_url,
                        crawl_session_id=task.crawl_session_id,
                        chunks=chunks,
                        document_metadata={
                            "url": page.url,
                            "depth": page.depth,
                            "ocr_action": ocr_action.value,
                        },
                        priority=task.priority,
                    )
                    
                    await self.queue_manager.put_storage_task(storage_task)
                    task.mark_completed(needs_ocr=False, chunks=len(chunks))
                    
                    logger.debug(
                        f"[{self.worker_id}] Created {len(chunks)} chunks, "
                        f"routed to storage: {page.url}"
                    )
                else:
                    # No chunks created (empty page)
                    task.mark_completed(needs_ocr=False, chunks=0)
                    logger.debug(f"[{self.worker_id}] No chunks created for {page.url}")
            
            return True
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to process {page.url}: {e}",
                exc_info=True
            )
            
            # Move to dead letter queue if max retries exceeded
            if task.retry_count >= task.max_retries:
                await self.queue_manager.put_dead_letter(
                    task,
                    f"Processing failed: {e}"
                )
            else:
                # Requeue for retry
                task.mark_failed(str(e))
                await self.queue_manager.put_process_task(task)
                logger.debug(
                    f"[{self.worker_id}] Requeued task "
                    f"(retry {task.retry_count}/{task.max_retries})"
                )
            
            return False
    
    async def _save_to_ocr_backlog(
        self,
        page: Page,
        ocr_action: OCRAction,
        website_url: str,
    ):
        """
        Save page to OCR backlog for later batch processing.
        
        Args:
            page: Page that needs OCR
            ocr_action: Type of OCR needed
            website_url: Website URL for grouping
        """
        try:
            # Save to JSON file (could also use MongoDB)
            backlog_dir = Path("outputs/ocr_backlog")
            backlog_dir.mkdir(parents=True, exist_ok=True)
            
            backlog_file = backlog_dir / "pending_ocr.jsonl"
            
            entry = {
                "url": page.url,
                "pdf_path": str(page.pdf_path) if page.pdf_path else None,
                "ocr_action": ocr_action.value,
                "website_url": website_url,
                "depth": page.depth,
                "timestamp": datetime.utcnow().isoformat(),
                "url_hash": page.url_hash if hasattr(page, 'url_hash') else None,
            }
            
            # Append to JSONL file
            with backlog_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            
            logger.info(f"[{self.worker_id}] Saved to OCR backlog: {page.url}")
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to save OCR backlog entry: {e}")
    
    async def _save_large_pdf_to_storage(
        self,
        page: Page,
        website_url: str,
        pdf_size_mb: float,
    ):
        """
        Save large PDF to separate storage folder and MongoDB without processing.
        
        Args:
            page: Page with large PDF
            website_url: Website URL for grouping
            pdf_size_mb: Size of PDF in MB
        """
        try:
            from app.config import LARGE_PDF_STORAGE_DIR
            from app.services.document_store import DocumentStore
            from app.schemas.document import DocumentStatus
            import shutil
            import uuid
            
            # Create large PDF storage directory
            storage_dir = Path(LARGE_PDF_STORAGE_DIR)
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy PDF to large files storage
            dest_path = storage_dir / page.pdf_path.name
            shutil.copy2(page.pdf_path, dest_path)
            
            # Save metadata to MongoDB
            store = DocumentStore.from_config()
            
            # Extract original filename from URL
            original_filename = page.url.split("/")[-1] or "document.pdf"
            
            await asyncio.to_thread(
                store.create_document,
                original_file=original_filename,
                source_url=page.url,
                file_path=str(dest_path),
                crawl_session_id=str(uuid.uuid4()),  # Could use crawl_session_id from task
                file_size=int(pdf_size_mb * 1024 * 1024),
                crawl_depth=page.depth if hasattr(page, 'depth') else 0,
                status=DocumentStatus.STORED,  # Mark as stored, not processed
            )
            
            logger.info(
                f"[{self.worker_id}] Saved large PDF to storage: {dest_path} "
                f"({pdf_size_mb:.2f} MB) - URL: {page.url}"
            )
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to save large PDF to storage: {e}",
                exc_info=True
            )