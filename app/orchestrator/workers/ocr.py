"""
OCR Worker

Performs OCR on pages using GPU (Surya models).
GPU-intensive worker - should have minimal concurrency (1-2 workers max).

⚠️ DEPRECATED: This worker uses old sentence-based chunking.
⚠️ TODO: Refactor to use Docling's HybridChunker like pdf_processor.py
⚠️ Currently disabled in orchestrator (ocr_workers=0)

If re-enabling OCR worker:
1. Remove chunk_page_text() calls
2. Convert OCR'd PDF to DoclingDocument
3. Use AsyncDocumentProcessor.create_chunks_async()
4. Follow pattern in app/orchestrator/workers/pdf_processor.py
"""

import asyncio
import time
from typing import Optional

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import OcrTask, StorageTask
from app.orchestrator.workers.helpers.ocr_processor import process_page_ocr
# REMOVED: chunk_page_text - needs refactoring to use HybridChunker
# from app.orchestrator.workers.helpers.text_processor import chunk_page_text
from app.crawling.models.document import PageContent, ContentSource

logger = get_logger(__name__)


class OcrWorker(BaseWorker):
    """
    OCR worker - handles GPU-based OCR processing.
    
    Responsibilities (GPU):
    - Pull OcrTask from ocr_queue
    - Run OCR using Surya (GPU)
    - Merge OCR text with existing content
    - Chunk the combined text
    - Push chunks to storage_queue
    
    IMPORTANT: Only 1-2 workers should run to prevent GPU overload!
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        min_chunk_words: int = 100,
        max_chunk_words: int = 512,
        overlap_words: int = 50,
    ):
        super().__init__(worker_id, queue_manager)
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        
        # OCR statistics
        self.total_ocr_time = 0.0
        self.pages_ocr_processed = 0
    
    @property
    def worker_type(self) -> str:
        return "ocr"
    
    async def startup(self):
        """Initialize OCR resources (loads GPU models)."""
        await super().startup()
        logger.info(f"[{self.worker_id}] OCR worker initialized (GPU)")
    
    async def get_next_task(self) -> Optional[OcrTask]:
        """Get next OCR task from queue."""
        return await self.queue_manager.get_ocr_task(timeout=2.0)
    
    async def process_task(self, task: OcrTask) -> bool:
        """
        Perform OCR on a page and prepare for storage.
        
        Args:
            task: OcrTask with page needing OCR
            
        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)
        
        page = task.page
        logger.info(f"[{self.worker_id}] Starting OCR for {page.url}")
        
        start_time = time.time()
        
        try:
            # Run OCR (GPU operation) with 60s timeout
            ocr_text = await asyncio.wait_for(
                process_page_ocr(page, task.ocr_action),
                timeout=60.0
            )
            
            processing_time = time.time() - start_time
            self.total_ocr_time += processing_time
            self.pages_ocr_processed += 1
            
            if ocr_text:
                # Merge OCR text with existing content
                if page.content and page.content.text:
                    # Combine DOM text + OCR text
                    combined_text = f"{page.content.text}\n\n{ocr_text}"
                    page.content = PageContent.from_text(combined_text, ContentSource.HYBRID)
                else:
                    # Only OCR text
                    page.content = PageContent.from_text(ocr_text, ContentSource.OCR)
                
                logger.info(
                    f"[{self.worker_id}] OCR completed for {page.url} "
                    f"({len(ocr_text)} chars in {processing_time:.2f}s)"
                )
            else:
                logger.warning(f"[{self.worker_id}] No OCR text extracted from {page.url}")
            
            # TODO: Refactor to use HybridChunker - chunk_page_text has been removed
            # This worker is currently disabled (ocr_workers=0 in config)
            logger.error(
                f"[{self.worker_id}] OCR worker needs refactoring to use HybridChunker. "
                f"Cannot process {page.url} - marking as failed"
            )
            chunks = []  # Stub: chunk_page_text was removed
            
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
                        "ocr_action": task.ocr_action.value,
                        "ocr_completed": True,
                        "processing_time_seconds": processing_time,
                    },
                    priority=task.priority,
                )
                
                await self.queue_manager.put_storage_task(storage_task)
                
                logger.info(
                    f"[{self.worker_id}] Created {len(chunks)} chunks "
                    f"from OCR'd page: {page.url}"
                )
            else:
                logger.warning(f"[{self.worker_id}] No chunks created from OCR'd page: {page.url}")
            
            # Mark task complete
            task.mark_completed(
                text_length=len(ocr_text) if ocr_text else 0,
                processing_time=processing_time
            )
            
            return True
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(
                f"[{self.worker_id}] OCR timeout (60s) for {page.url} - skipping OCR, "
                f"chunking with existing content"
            )
            
            # TODO: Refactor to use HybridChunker - chunk_page_text has been removed
            logger.error(
                f"[{self.worker_id}] OCR worker needs refactoring to use HybridChunker. "
                f"Cannot process {page.url} - marking as failed"
            )
            chunks = []  # Stub: chunk_page_text was removed
            
            if chunks:
                storage_task = StorageTask(
                    task_id=f"{task.task_id}_storage",
                    website_url=task.website_url,
                    crawl_session_id=task.crawl_session_id,
                    chunks=chunks,
                    document_metadata={
                        "url": page.url,
                        "ocr_skipped": "timeout_60s",
                        "processing_time_seconds": processing_time,
                    },
                    priority=task.priority,
                )
                await self.queue_manager.put_storage_task(storage_task)
                logger.info(
                    f"[{self.worker_id}] Created {len(chunks)} chunks "
                    f"(no OCR due to timeout): {page.url}"
                )
            
            task.mark_completed()
            return True  # Don't fail the task, just skip OCR
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"[{self.worker_id}] OCR failed for {page.url} "
                f"after {processing_time:.2f}s: {e}",
                exc_info=True
            )
            
            # Move to dead letter queue if max retries exceeded
            if task.retry_count >= task.max_retries:
                await self.queue_manager.put_dead_letter(
                    task,
                    f"OCR failed: {e}"
                )
            else:
                # Requeue for retry (but OCR is expensive, so limit retries)
                task.mark_failed(str(e))
                await self.queue_manager.put_ocr_task(task)
                logger.info(
                    f"[{self.worker_id}] Requeued OCR task "
                    f"(retry {task.retry_count}/{task.max_retries})"
                )
            
            return False
    
    @property
    def stats(self) -> dict:
        """Get OCR worker statistics including GPU metrics."""
        base_stats = super().stats
        
        avg_ocr_time = (
            self.total_ocr_time / self.pages_ocr_processed
            if self.pages_ocr_processed > 0 else 0
        )
        
        base_stats.update({
            "pages_ocr_processed": self.pages_ocr_processed,
            "total_ocr_time_seconds": self.total_ocr_time,
            "avg_ocr_time_seconds": avg_ocr_time,
        })
        
        return base_stats
