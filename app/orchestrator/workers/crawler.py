"""
Crawler Worker

Crawls websites and pushes discovered pages to the processing queue.
"""

import uuid
from typing import Optional
from datetime import datetime

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import CrawlTask, ProcessTask, TaskPriority
from app.crawling.stages.crawler import CrawlerStage
from app.crawling.models.document import Document, Page
from app.crawling.models.config import PipelineConfig

logger = get_logger(__name__)


class CrawlerWorker(BaseWorker):
    """
    Crawler worker - handles website crawling.
    
    Responsibilities:
    - Pull CrawlTask from crawl_queue
    - Crawl website using CrawlerStage
    - Push individual pages to processing_queue (streaming as discovered)
    - Update MongoDB with crawl status
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(worker_id, queue_manager)
        self.config = config or PipelineConfig()
        self.crawler_stage = CrawlerStage(self.config)
        self._current_task: Optional[CrawlTask] = None
        self._pages_queued = 0
    
    @property
    def worker_type(self) -> str:
        return "crawler"
    
    async def startup(self):
        """Initialize crawler resources."""
        await super().startup()
        await self.crawler_stage.setup()
        logger.info(f"[{self.worker_id}] Crawler initialized")
    
    async def shutdown(self):
        """Cleanup crawler resources."""
        await self.crawler_stage.teardown()
        await super().shutdown()
    
    async def get_next_task(self) -> Optional[CrawlTask]:
        """Get next crawl task from queue."""
        return await self.queue_manager.get_crawl_task(timeout=1.0)
    
    async def _on_page_crawled(self, page: Page) -> None:
        """
        Callback when a page is successfully crawled.
        Immediately queues the page for processing (streaming approach).
        
        Args:
            page: The crawled page with content
        """
        if not self._current_task:
            return
        
        # Only queue pages with content
        if page.html_content or page.pdf_path or page.dom_text:
            process_task = ProcessTask(
                task_id=f"{self._current_task.task_id}_page_{page.url_hash}",
                page=page,
                website_url=self._current_task.website_url,
                crawl_session_id=self._current_task.crawl_session_id,
                priority=self._current_task.priority,
            )
            
            await self.queue_manager.put_process_task(process_task)
            self._pages_queued += 1
            logger.info(
                f"[{self.worker_id}] ✓ Queued page {self._pages_queued}: {page.url[:60]}"
            )
    
    async def process_task(self, task: CrawlTask) -> bool:
        """
        Crawl a website and push pages to processing queue (streaming).
        
        Pages are queued immediately as they're discovered, allowing
        downstream workers to start processing while crawling continues.
        
        Args:
            task: CrawlTask with website URL
            
        Returns:
            True if successful
        """
        # Defensive check: ensure task is correct type
        if not isinstance(task, CrawlTask):
            logger.error(
                f"[{self.worker_id}] Received wrong task type: {type(task).__name__}. "
                f"Expected CrawlTask. Skipping task."
            )
            return False
        
        task.mark_started(self.worker_id)
        self._current_task = task
        self._pages_queued = 0
        
        logger.info(
            f"[{self.worker_id}] Crawling {task.website_url} "
            f"(max_pages={task.max_pages}, max_depth={task.max_depth})"
        )
        
        try:
            # Create document for crawling
            document = Document(
                start_url=task.website_url,
                output_dir=self.config.output_dir,
                crawl_depth=task.max_depth,
                max_pages=task.max_pages,
            )
            
            # Set callback for streaming pages
            self.crawler_stage.on_page_crawled = self._on_page_crawled
            
            # Run crawler (will stream pages to processing queue via callback)
            document = await self.crawler_stage.process(document)
            
            # Update task progress
            task.pages_discovered = len(document.pages)
            task.pages_crawled = len([p for p in document.pages if p.html_content or p.pdf_path])
            
            logger.info(
                f"[{self.worker_id}] Crawl complete: {task.pages_crawled}/{task.pages_discovered} pages, "
                f"{self._pages_queued} queued for processing"
            )
            
            # Mark task complete
            task.mark_completed()
            self._current_task = None
            return True
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to crawl {task.website_url}: {e}",
                exc_info=True
            )
            
            # Move to dead letter queue if max retries exceeded
            if task.retry_count >= task.max_retries:
                await self.queue_manager.put_dead_letter(
                    task,
                    f"Max retries exceeded: {e}"
                )
            else:
                # Requeue for retry
                task.mark_failed(str(e))
                await self.queue_manager.put_crawl_task(task)
                logger.info(f"[{self.worker_id}] Requeued task (retry {task.retry_count}/{task.max_retries})")
            
            return False
