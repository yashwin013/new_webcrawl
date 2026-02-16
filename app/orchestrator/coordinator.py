"""
Multi-Site Orchestrator Coordinator

Coordinates parallel crawling of multiple websites with automatic resource management.
"""

import asyncio
from typing import List, Dict, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

from app.config import get_logger
from app.orchestrator.config import OrchestratorConfig, get_default_config
from app.orchestrator.queues import QueueManager
from app.orchestrator.workers import (
    CrawlerWorker,
    ProcessorWorker,
    OcrWorker,
    StorageWorker,
)
from app.orchestrator.workers.pdf_processor import PdfProcessorWorker
from app.orchestrator.workers.recovery import WorkerHealthMonitor, WorkerTimeout
from app.orchestrator.models.task import CrawlTask, TaskPriority
from app.orchestrator.monitoring import (
    HealthMonitor,
    ProgressTracker,
    OrchestratorHealth,
)

logger = get_logger(__name__)


@dataclass
class OrchestratorStats:
    """Overall orchestrator statistics."""
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Websites
    total_websites: int = 0
    websites_completed: int = 0
    websites_failed: int = 0
    websites_in_progress: int = 0
    
    # Workers
    active_workers: int = 0
    total_workers: int = 0
    
    # Tasks
    total_tasks_processed: int = 0
    total_tasks_failed: int = 0
    
    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    @property
    def websites_remaining(self) -> int:
        """Number of websites remaining."""
        return self.total_websites - self.websites_completed - self.websites_failed
    
    @property
    def completion_percent(self) -> float:
        """Overall completion percentage."""
        if self.total_websites == 0:
            return 0.0
        completed = self.websites_completed + self.websites_failed
        return (completed / self.total_websites) * 100


class MultiSiteOrchestrator:
    """
    Orchestrates parallel crawling of multiple websites.
    
    Manages the complete pipeline:
    - Spawns worker pools (crawlers, processors, OCR, storage)
    - Feeds websites into the crawl queue
    - Monitors progress and health
    - Handles graceful shutdown
    
    Usage:
        orchestrator = MultiSiteOrchestrator(config)
        await orchestrator.startup()
        await orchestrator.crawl_websites(["https://site1.com", "https://site2.com"])
        await orchestrator.shutdown()
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize orchestrator with configuration."""
        self.config = config or get_default_config()
        
        # Core components
        self.queue_manager: Optional[QueueManager] = None
        
        # Worker pools
        self.crawler_workers: List[CrawlerWorker] = []
        self.processor_workers: List[ProcessorWorker] = []
        self.pdf_workers: List[PdfProcessorWorker] = []
        self.ocr_workers: List[OcrWorker] = []
        self.storage_workers: List[StorageWorker] = []
        
        # Worker tasks (for lifecycle management)
        self.worker_tasks: List[asyncio.Task] = []
        
        # Website tracking
        self.websites: List[str] = []
        self.websites_completed: Set[str] = set()
        self.websites_failed: Set[str] = set()
        self.websites_in_progress: Set[str] = set()
        
        # State
        self.is_running: bool = False
        self.stats = OrchestratorStats()
        
        # Monitoring
        self.health_monitor: Optional[HealthMonitor] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Worker recovery
        self.worker_recovery: Optional[WorkerHealthMonitor] = None
        
        # Shutdown coordination
        self._shutdown_event: Optional[asyncio.Event] = None
    
    async def startup(self):
        """Initialize all components and spawn worker pools."""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("="*70)
        logger.info("Starting Multi-Site Orchestrator")
        logger.info("="*70)
        
        # Initialize queue manager
        self.queue_manager = QueueManager(self.config.queues)
        await self.queue_manager.startup()
        
        # Create shutdown event
        self._shutdown_event = asyncio.Event()
        
        # Spawn worker pools
        await self._spawn_workers()
        
        # Initialize monitoring
        self.health_monitor = HealthMonitor(self)
        self.progress_tracker = ProgressTracker(self)
        
        # Initialize worker recovery system
        if self.config.recovery.enable_recovery:
            logger.info("Initializing worker recovery system...")
            
            # Create timeouts from config
            timeouts = WorkerTimeout(
                crawler=self.config.recovery.crawler_timeout,
                processor=self.config.recovery.processor_timeout,
                pdf=self.config.recovery.pdf_timeout,
                ocr=self.config.recovery.ocr_timeout,
                storage=self.config.recovery.storage_timeout,
            )
            
            self.worker_recovery = WorkerHealthMonitor(
                timeouts=timeouts,
                max_task_retries=self.config.recovery.max_task_retries,
                check_interval=self.config.recovery.check_interval,
            )
            
            # Start recovery monitoring
            await self.worker_recovery.start_monitoring(
                get_workers_callback=self._get_all_workers,
                restart_worker_callback=self._restart_worker,
                requeue_task_callback=self._requeue_task,
            )
            logger.info("✓ Worker recovery system active")
        else:
            logger.info("Worker recovery system disabled")
            self.worker_recovery = None
        
        # Start monitoring task if enabled
        if self.config.enable_monitoring:
            self._monitoring_task = asyncio.create_task(self._run_monitoring())
        
        # Update stats
        self.stats.total_workers = len(self.worker_tasks)
        self.stats.active_workers = len(self.worker_tasks)
        
        self.is_running = True
        
        logger.info("="*70)
        logger.info(f"✓ Orchestrator started with {self.stats.total_workers} workers")
        if self.config.enable_monitoring:
            logger.info("✓ Monitoring enabled")
        if self.config.recovery.enable_recovery:
            logger.info("✓ Worker recovery system enabled")
        logger.info(self.config.summary)
        logger.info("="*70)
    
    async def _spawn_workers(self):
        """Spawn all worker pools."""
        logger.info("Spawning worker pools...")
        
        # Spawn crawler workers
        for i in range(self.config.workers.crawler_workers):
            worker = CrawlerWorker(f"crawler-{i+1}", self.queue_manager)
            await worker.startup()
            self.crawler_workers.append(worker)
            
            # Start worker task
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logger.info(f"  ✓ {len(self.crawler_workers)} crawler workers")
        
        # Spawn processor workers
        enable_ocr = self.config.workers.ocr_workers > 0
        for i in range(self.config.workers.processor_workers):
            worker = ProcessorWorker(
                f"processor-{i+1}",
                self.queue_manager,
                enable_ocr=enable_ocr,  # Disable OCR routing if no OCR workers
            )
            await worker.startup()
            self.processor_workers.append(worker)
            
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logger.info(f"  ✓ {len(self.processor_workers)} processor workers")
        
        # Spawn PDF processor workers (GPU - Docling)
        for i in range(self.config.workers.pdf_workers):
            worker = PdfProcessorWorker(f"pdf-{i+1}", self.queue_manager)
            await worker.startup()
            self.pdf_workers.append(worker)
            
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logger.info(f"  ✓ {len(self.pdf_workers)} PDF processor workers (GPU - Docling)")
        
        # Spawn OCR workers
        if self.config.workers.ocr_workers == 0:
            logger.info("  ⚠ OCR disabled (0 OCR workers) - scanned PDFs saved to backlog for Phase 2")
        
        for i in range(self.config.workers.ocr_workers):
            worker = OcrWorker(f"ocr-{i+1}", self.queue_manager)
            await worker.startup()
            self.ocr_workers.append(worker)
            
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logger.info(f"  ✓ {len(self.ocr_workers)} OCR workers (GPU)")
        
        # Spawn storage workers
        for i in range(self.config.workers.storage_workers):
            worker = StorageWorker(
                f"storage-{i+1}",
                self.queue_manager,
                store_to_qdrant=False,  # Can be configured
            )
            await worker.startup()
            self.storage_workers.append(worker)
            
            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)
        
        logger.info(f"  ✓ {len(self.storage_workers)} storage workers")
    
    async def crawl_websites(
        self,
        website_urls: List[str],
        max_pages_per_site: int = 50,
        max_depth: int = 3,
        crawl_session_id: Optional[str] = None,
    ):
        """
        Crawl multiple websites in parallel.
        
        Args:
            website_urls: List of website URLs to crawl
            max_pages_per_site: Maximum pages per website
            max_depth: Maximum crawl depth
            crawl_session_id: Optional session ID for tracking
        """
        if not self.is_running:
            raise RuntimeError("Orchestrator not started. Call startup() first.")
        
        if not website_urls:
            logger.warning("No websites provided")
            return
        
        # Generate session ID if not provided
        if not crawl_session_id:
            crawl_session_id = f"session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        self.websites = website_urls
        self.stats.total_websites = len(website_urls)
        
        logger.info("="*70)
        logger.info(f"Starting crawl session: {crawl_session_id}")
        logger.info(f"Websites to crawl: {len(website_urls)}")
        logger.info("="*70)
        
        # Check if any URLs have already been crawled
        from app.services.document_store import DocumentStore
        document_store = DocumentStore.from_config()
        
        urls_to_crawl = []
        already_crawled = []
        
        for url in website_urls:
            is_crawled, details = await asyncio.to_thread(
                document_store.is_url_already_crawled, url
            )
            if is_crawled:
                already_crawled.append((url, details))
                logger.info(
                    f"⚠ SKIPPING {url} - Already crawled on "
                    f"{details.get('crawled_at', 'unknown date')} "
                    f"(Session: {details.get('crawl_session_id', 'N/A')[:8]}...)"
                )
            else:
                urls_to_crawl.append(url)
                logger.info(f"✓ Queuing {url} for crawling")
        
        # Report results
        if already_crawled:
            logger.info("="*70)
            logger.info(f" Duplicate Detection Summary:")
            logger.info(f"   Total URLs submitted: {len(website_urls)}")
            logger.info(f"   Already crawled: {len(already_crawled)}")
            logger.info(f"   New URLs to crawl: {len(urls_to_crawl)}")
            logger.info("="*70)
            
            if already_crawled:
                logger.info("\n Previously Crawled URLs:")
                for url, details in already_crawled:
                    logger.info(f"   • {url}")
                    logger.info(f"     Session: {details.get('crawl_session_id', 'N/A')}")
                    logger.info(f"     Crawled: {details.get('crawled_at', 'N/A')}")
                    if details.get('is_fully_crawled'):
                        logger.info(f"     Status: ✓ Fully crawled")
                    else:
                        logger.info(
                            f"     Status: Partial ({details.get('crawled_documents', 0)}/"
                            f"{details.get('total_documents', 0)} documents)"
                        )
                logger.info("="*70)
        
        # If all URLs were already crawled, exit early
        if not urls_to_crawl:
            logger.warning("⚠ All URLs have already been crawled. Nothing to do.")
            self.stats.websites_completed = len(already_crawled)
            return
        
        # Update stats with only new URLs
        self.websites = urls_to_crawl
        self.stats.total_websites = len(urls_to_crawl)
        
        # Create crawl tasks for each website
        for idx, url in enumerate(urls_to_crawl):
            task = CrawlTask(
                task_id=f"{crawl_session_id}-crawl-{idx+1}",
                website_url=url,
                crawl_session_id=crawl_session_id,
                max_pages=max_pages_per_site,
                max_depth=max_depth,
                priority=TaskPriority.NORMAL,
            )
            
            # Add to crawl queue
            await self.queue_manager.put_crawl_task(task)
            self.websites_in_progress.add(url)
            
            logger.info(f"  → Queued: {url}")
        
        logger.info("="*70)
        logger.info("All websites queued. Workers processing...")
        logger.info("="*70)
        
        # Monitor progress (optional - can be done separately)
        if self.config.enable_monitoring:
            await self._monitor_progress()
    
    async def _monitor_progress(self):
        """Monitor progress until all websites are complete."""
        logger.info("\nMonitoring progress...")
        
        # Grace period: Give workers time to start and grab tasks
        # Wait 15 seconds before any completion checks
        logger.info("Waiting 15 seconds for workers to start processing...")
        
        # Check for shutdown during grace period (in 1-second intervals)
        for _ in range(15):
            if self._shutdown_event.is_set():
                logger.info("Shutdown signal received during grace period")
                return
            await asyncio.sleep(1)
        
        logger.info("Grace period complete. Monitoring work progress...")
        
        # Track last progress to detect deadlocks
        last_processed_count = 0
        stalled_iterations = 0
        max_stalled_iterations = 20  # 20 * 5 seconds = 100 seconds without progress
        
        while not self._shutdown_event.is_set():
            # Check if all queues are empty and workers are idle
            if await self._is_work_complete():
                logger.info("All work complete!")
                break
            
            # Check for deadlock: if nothing has been processed for too long
            current_processed = self.stats.total_tasks_processed
            
            # Count active workers to avoid false stall warnings
            all_workers = (
                self.crawler_workers +
                self.processor_workers +
                self.pdf_workers +
                self.ocr_workers +
                self.storage_workers
            )
            busy_workers_count = sum(1 for worker in all_workers if worker.is_busy)
            
            # Only warn about stall if no progress AND no workers are busy
            if current_processed == last_processed_count:
                if busy_workers_count == 0:
                    # Truly stalled - no progress and no workers working
                    stalled_iterations += 1
                    if stalled_iterations >= max_stalled_iterations:
                        logger.warning(
                            f"⚠ Pipeline appears stalled! No progress for {stalled_iterations * self.config.monitoring_interval_seconds}s"
                        )
                        logger.warning("This may indicate hung workers or blocked queues.")
                        logger.warning("Continuing to wait, but consider restarting if this persists...")
                else:
                    # Workers are busy, so this is normal - reset counter
                    stalled_iterations = 0
            else:
                stalled_iterations = 0  # Reset counter if progress was made
                last_processed_count = current_processed
            
            # Log progress
            await self._log_progress()
            
            # Wait before next check (check shutdown every second)
            for _ in range(self.config.monitoring_interval_seconds):
                if self._shutdown_event.is_set():
                    logger.info("Shutdown signal received during monitoring")
                    return
                await asyncio.sleep(1)
    
    async def _is_work_complete(self) -> bool:
        """Check if all work is complete."""
        # Check if all queues are empty
        if not self.queue_manager:
            return False
        
        crawl_size = self.queue_manager.crawl_queue.qsize()
        processing_size = self.queue_manager.processing_queue.qsize()
        ocr_size = self.queue_manager.ocr_queue.qsize()
        storage_size = self.queue_manager.storage_queue.qsize()
        
        all_empty = (
            crawl_size == 0 and
            processing_size == 0 and
            ocr_size == 0 and
            storage_size == 0
        )
        
        if not all_empty:
            logger.debug(
                f"Queues not empty: crawl={crawl_size}, processing={processing_size}, "
                f"ocr={ocr_size}, storage={storage_size}"
            )
            return False
        
        # Check if all workers are idle (no active tasks)
        all_workers = (
            self.crawler_workers +
            self.processor_workers +
            self.pdf_workers +
            self.ocr_workers +
            self.storage_workers
        )
        
        # Count busy workers for debugging
        busy_workers = [worker for worker in all_workers if worker.is_busy]
        if busy_workers:
            logger.info(
                f"Queues empty but {len(busy_workers)} workers still busy: "
                f"{[w.worker_id for w in busy_workers]}"
            )
        
        any_busy = any(worker.is_busy for worker in all_workers)
        
        # Work is complete only if queues are empty AND no workers are busy
        is_complete = all_empty and not any_busy
        if is_complete:
            logger.info("All queues empty and all workers idle - work complete!")
        
        return is_complete
    
    async def _log_progress(self):
        """Log current progress."""
        metrics = self.queue_manager.get_metrics()
        
        logger.info("\n" + "="*70)
        logger.info(f"Progress Report (Elapsed: {self.stats.duration_seconds:.1f}s)")
        logger.info("-"*70)
        
        # Queue status
        logger.info("Queue Status:")
        for name, metric in metrics.items():
            if name != "dead_letter":
                logger.info(
                    f"  {name:12s}: {metric.current_size:3d}/{metric.max_size:3d} "
                    f"({metric.utilization_percent:5.1f}%) | "
                    f"Added: {metric.total_added:4d} | "
                    f"Removed: {metric.total_removed:4d}"
                )
        
        # Dead letter queue
        dlq = metrics.get("dead_letter")
        if dlq and dlq.total_failed > 0:
            logger.info(f"  Dead Letter : {dlq.current_size:3d} failed tasks")
        
        # Worker statistics
        logger.info("\nWorker Statistics:")
        self._log_worker_stats("Crawlers", self.crawler_workers)
        self._log_worker_stats("Processors", self.processor_workers)
        self._log_worker_stats("OCR", self.ocr_workers)
        self._log_worker_stats("Storage", self.storage_workers)
        
        logger.info("="*70)
    
    def _log_worker_stats(self, name: str, workers: List):
        """Log statistics for a worker pool."""
        if not workers:
            return
        
        total_processed = sum(w.tasks_processed for w in workers)
        total_failed = sum(w.tasks_failed for w in workers)
        
        logger.info(
            f"  {name:12s}: {len(workers)} workers | "
            f"Processed: {total_processed:4d} | "
            f"Failed: {total_failed:3d}"
        )
    
    def _get_all_workers(self) -> List:
        """Get all active workers for health monitoring."""
        return (
            self.crawler_workers +
            self.processor_workers +
            self.pdf_workers +
            self.ocr_workers +
            self.storage_workers
        )
    
    def _restart_worker(self, old_worker) -> asyncio.Task:
        """
        Restart a worker by creating a new instance and starting it.
        
        Args:
            old_worker: The worker that needs to be restarted
            
        Returns:
            New worker task
        """
        worker_id = old_worker.worker_id
        worker_type = old_worker.worker_type
        
        logger.info(f"Creating replacement worker for {worker_id} ({worker_type})")
        
        # Create new worker based on type
        new_worker = None
        worker_list = None
        
        if worker_type == "crawler":
            new_worker = CrawlerWorker(worker_id, self.queue_manager)
            worker_list = self.crawler_workers
        elif worker_type == "processor":
            enable_ocr = self.config.workers.ocr_workers > 0
            new_worker = ProcessorWorker(worker_id, self.queue_manager, enable_ocr=enable_ocr)
            worker_list = self.processor_workers
        elif worker_type == "pdf":
            new_worker = PdfProcessorWorker(worker_id, self.queue_manager)
            worker_list = self.pdf_workers
        elif worker_type == "ocr":
            new_worker = OcrWorker(worker_id, self.queue_manager)
            worker_list = self.ocr_workers
        elif worker_type == "storage":
            new_worker = StorageWorker(worker_id, self.queue_manager, store_to_qdrant=False)
            worker_list = self.storage_workers
        
        if new_worker and worker_list is not None:
            # Remove old worker from list
            try:
                worker_list.remove(old_worker)
            except ValueError:
                pass
            
            # Add new worker to list
            worker_list.append(new_worker)
            
            # Start the new worker
            async def start_new_worker():
                await new_worker.startup()
                await new_worker.run()
            
            new_task = asyncio.create_task(start_new_worker())
            
            # Replace in worker tasks list
            if old_worker._task in self.worker_tasks:
                try:
                    self.worker_tasks.remove(old_worker._task)
                except ValueError:
                    pass
            self.worker_tasks.append(new_task)
            
            return new_task
        
        logger.error(f"Failed to create replacement for unknown worker type: {worker_type}")
        return None
    
    async def _requeue_task(self, task):
        """
        Requeue a failed task to its appropriate queue.
        
        Args:
            task: Task object to requeue
        """
        # Determine task type and route to appropriate queue
        task_type = type(task).__name__
        
        try:
            if hasattr(task, 'website_url'):
                # Crawl task
                await self.queue_manager.put_crawl_task(task)
                logger.debug(f"Requeued crawl task: {task.website_url}")
            elif hasattr(task, 'url') and hasattr(task, 'html_content'):
                # Processing task
                await self.queue_manager.put_processing_task(task)
                logger.debug(f"Requeued processing task: {task.url}")
            elif hasattr(task, 'pdf_url') or (hasattr(task, 'url') and 'pdf' in str(task.url).lower()):
                # PDF task
                await self.queue_manager.put_pdf_task(task)
                logger.debug(f"Requeued PDF task")
            elif hasattr(task, 'chunks'):
                # Storage task
                await self.queue_manager.put_storage_task(task)
                logger.debug(f"Requeued storage task")
            else:
                logger.warning(f"Unknown task type for requeue: {task_type}")
        except Exception as e:
            logger.error(f"Failed to requeue task: {e}", exc_info=True)
    
    async def shutdown(self, timeout: Optional[float] = None):
        """
        Gracefully shutdown orchestrator.
        
        Args:
            timeout: Timeout in seconds (uses config default if not provided)
        """
        if not self.is_running:
            logger.warning("Orchestrator not running")
            return
        
        timeout = timeout or self.config.shutdown_timeout_seconds
        
        logger.info("="*70)
        logger.info("Shutting down orchestrator...")
        logger.info("="*70)
        
        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()
        
        # Stop worker recovery
        if self.worker_recovery:
            logger.info("Stopping worker recovery...")
            await self.worker_recovery.stop_monitoring()
            
            # Log recovery stats
            recovery_stats = self.worker_recovery.get_stats()
            if recovery_stats.total_recoveries > 0:
                logger.info(f"Recovery stats: {recovery_stats}")
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all workers
        logger.info("Shutting down workers...")
        all_workers = (
            self.crawler_workers +
            self.processor_workers +
            self.pdf_workers +
            self.ocr_workers +
            self.storage_workers
        )
        
        for worker in all_workers:
            await worker.shutdown()
        
        # Cancel all worker tasks
        logger.info("Cancelling worker tasks...")
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish with timeout
        if self.worker_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.worker_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Worker shutdown timed out after {timeout}s")
        
        # Shutdown queue manager
        if self.queue_manager:
            await self.queue_manager.shutdown()
        
        # Update stats
        self.stats.end_time = datetime.utcnow()
        self.is_running = False
        
        # Final report
        logger.info("="*70)
        logger.info("Orchestrator Shutdown Complete")
        logger.info("-"*70)
        logger.info(f"Duration: {self.stats.duration_seconds:.1f}s")
        logger.info(f"Websites: {self.stats.websites_completed} completed, {self.stats.websites_failed} failed")
        logger.info("="*70)
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.startup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.shutdown()
        return False
    
    def get_stats(self) -> Dict:
        """Get current orchestrator statistics."""
        return {
            "orchestrator": {
                "is_running": self.is_running,
                "duration_seconds": self.stats.duration_seconds,
                "completion_percent": self.stats.completion_percent,
            },
            "websites": {
                "total": self.stats.total_websites,
                "completed": self.stats.websites_completed,
                "failed": self.stats.websites_failed,
                "in_progress": self.stats.websites_in_progress,
                "remaining": self.stats.websites_remaining,
            },
            "workers": {
                "total": self.stats.total_workers,
                "active": self.stats.active_workers,
                "crawlers": len(self.crawler_workers),
                "processors": len(self.processor_workers),
                "ocr": len(self.ocr_workers),
                "storage": len(self.storage_workers),
            },
            "tasks": {
                "processed": self.stats.total_tasks_processed,
                "failed": self.stats.total_tasks_failed,
            },
        }
    
    async def _run_monitoring(self):
        """Background task for periodic monitoring."""
        try:
            while not self._shutdown_event.is_set():
                # Get health status
                if self.health_monitor:
                    health = self.health_monitor.get_overall_health()
                    
                    # Log if unhealthy or has alerts
                    if not health.is_healthy or health.alerts:
                        self.health_monitor.log_health_report(health)
                
                # Get progress report
                if self.progress_tracker:
                    report = self.progress_tracker.get_progress_report()
                    self.progress_tracker.log_progress_report(report)
                
                # Wait before next check
                await asyncio.sleep(self.config.monitoring_interval_seconds)
        
        except asyncio.CancelledError:
            logger.debug("Monitoring task cancelled")
        except Exception as e:
            logger.error(f"Monitoring error: {e}", exc_info=True)
    
    def get_health(self) -> Optional[OrchestratorHealth]:
        """Get current health status."""
        if self.health_monitor:
            return self.health_monitor.get_overall_health()
        return None
    
    def get_progress(self) -> Optional[Dict]:
        """Get current progress report."""
        if self.progress_tracker:
            return self.progress_tracker.get_progress_report()
        return None
