"""
PDF Processor Worker

Receives PDF tasks from the crawl queue and saves them to MongoDB.
Actual vectorization (Docling HybridChunker + Qdrant) is handled
post-crawl by vectorize_crawled_pdfs() in app/docling/pipeline.py.
"""

import asyncio
import time
import json
from typing import Optional
from pathlib import Path

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import PdfTask
from app.crawling.models.document import Page

logger = get_logger(__name__)


class PdfProcessorWorker(BaseWorker):
    """
    PDF processor worker - receives PDF tasks and records them for post-crawl vectorization.

    PDF vectorization (Docling + HybridChunker + Qdrant) is handled automatically
    after the crawl completes via vectorize_crawled_pdfs() in app/docling/pipeline.py.

    This worker's role:
    - Pull PdfTask from pdf_queue
    - Acknowledge the task (PDF already saved to disk + MongoDB by CrawlerStage)
    - If PDF is scanned/minimal text: save to OCR backlog for Phase 2
    """

    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        timeout_seconds: int = 120,
    ):
        super().__init__(worker_id, queue_manager)
        self.timeout_seconds = timeout_seconds

        # Statistics
        self.pdfs_processed = 0
        self.pdfs_failed = 0

    @property
    def worker_type(self) -> str:
        return "pdf_processor"

    async def startup(self):
        """Initialize PDF processor worker."""
        await super().startup()
        logger.info(f"[{self.worker_id}] PDF processor worker initialized")
        logger.info(f"[{self.worker_id}] Note: Vectorization handled post-crawl by Process_Docling pipeline")

    async def get_next_task(self) -> Optional[PdfTask]:
        """Get next PDF task from queue."""
        return await self.queue_manager.get_pdf_task(timeout=2.0)

    async def process_task(self, task: PdfTask) -> bool:
        """
        Acknowledge a PDF task.

        The PDF has already been downloaded to disk and saved to MongoDB
        by CrawlerStage._download_pdf(). Vectorization will happen automatically
        after the crawl via vectorize_crawled_pdfs().

        Args:
            task: PdfTask with PDF metadata

        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)

        page = task.page
        pdf_path = page.pdf_path

        if not pdf_path or not pdf_path.exists():
            logger.error(f"[{self.worker_id}] PDF not found: {pdf_path}")
            task.mark_failed("PDF file not found")
            self.pdfs_failed += 1
            return False

        logger.info(
            f"[{self.worker_id}] PDF queued for post-crawl vectorization: {page.url} "
            f"({pdf_path.stat().st_size // 1024} KB)"
        )

        self.pdfs_processed += 1
        task.mark_completed()
        return True

    async def _save_to_ocr_backlog(self, page: Page, website_url: str):
        """Save page to OCR backlog for Phase 2 processing."""
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
            "url_hash": page.url_hash if hasattr(page, "url_hash") else None,
            "reason": "scanned_pdf_needs_ocr",
        }

        with backlog_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"[{self.worker_id}] Saved to OCR backlog: {page.url}")

    @property
    def stats(self) -> dict:
        """Get PDF processor worker statistics."""
        base_stats = super().stats
        base_stats.update({
            "pdfs_processed": self.pdfs_processed,
            "pdfs_failed": self.pdfs_failed,
        })
        return base_stats
