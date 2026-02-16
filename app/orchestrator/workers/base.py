"""
Base Worker Class

Provides common functionality for all worker types in the orchestrator.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from app.config import get_logger
from app.orchestrator.queues import QueueManager
from app.core.backoff import ExponentialBackoff

logger = get_logger(__name__)


class BaseWorker(ABC):
    """
    Base class for all workers in the orchestrator.
    
    Provides common patterns:
    - Worker lifecycle (startup/shutdown)
    - Queue management
    - Error handling
    - Graceful cancellation
    
    Subclasses must implement:
    - process_task(): Main work logic
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
    ):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique identifier for this worker
            queue_manager: Shared queue manager
        """
        self.worker_id = worker_id
        self.queue_manager = queue_manager
        
        # Worker state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._is_busy = False  # Track if actively processing a task
        
        # Current task tracking (for hang detection)
        self.current_task = None
        self.current_task_start_time: Optional[datetime] = None
        
        # Statistics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.start_time: Optional[datetime] = None
        
        # Exponential backoff for consecutive errors
        self._error_backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0)
        self._consecutive_errors = 0
    
    @property
    def is_busy(self) -> bool:
        """Check if worker is actively processing a task."""
        return self._is_busy
    
    @property
    @abstractmethod
    def worker_type(self) -> str:
        """Worker type name (crawler, processor, ocr, storage)."""
        pass
    
    @abstractmethod
    async def process_task(self, task) -> bool:
        """
        Process a single task.
        
        Args:
            task: Task object to process
            
        Returns:
            True if successful, False if failed
        """
        pass
    
    @abstractmethod
    async def get_next_task(self) -> Optional[any]:
        """
        Get next task from appropriate queue.
        
        Returns:
            Task object or None if no task available
        """
        pass
    
    async def startup(self):
        """Initialize worker resources."""
        logger.info(f"[{self.worker_id}] Starting {self.worker_type} worker")
        self.start_time = datetime.utcnow()
        self._running = True
    
    async def shutdown(self):
        """Cleanup worker resources."""
        logger.info(f"[{self.worker_id}] Shutting down...")
        self._running = False
        self._shutdown_event.set()
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(
            f"[{self.worker_id}] Shutdown complete "
            f"(processed: {self.tasks_processed}, failed: {self.tasks_failed})"
        )
    
    async def run(self):
        """
        Main worker loop.
        
        Continuously processes tasks until shutdown is requested.
        """
        await self.startup()
        
        logger.info(f"[{self.worker_id}] Worker loop started")
        
        try:
            while self._running and not self.queue_manager.is_shutdown_requested:
                try:
                    # Get next task with timeout
                    task = await self.get_next_task()
                    
                    if task is None:
                        # No task available, wait a bit
                        await asyncio.sleep(0.5)
                        self._consecutive_errors = 0  # Reset on idle
                        continue
                    
                    # Mark as busy before processing
                    self._is_busy = True
                    self.current_task = task
                    self.current_task_start_time = datetime.utcnow()
                    try:
                        # Process task
                        success = await self.process_task(task)
                        
                        if success:
                            self.tasks_processed += 1
                            self._consecutive_errors = 0  # Reset on success
                        else:
                            self.tasks_failed += 1
                            self._consecutive_errors += 1
                    finally:
                        # Mark as idle after processing
                        self._is_busy = False
                        self.current_task = None
                        self.current_task_start_time = None
                        
                except asyncio.CancelledError:
                    logger.info(f"[{self.worker_id}] Cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"[{self.worker_id}] Unexpected error: {e}", exc_info=True)
                    self.tasks_failed += 1
                    self._consecutive_errors += 1
                    
                    # Use exponential backoff on consecutive errors
                    await self._error_backoff.wait(self._consecutive_errors - 1)
                    
        finally:
            await self.shutdown()
    
    def start(self) -> asyncio.Task:
        """
        Start worker as an async task.
        
        Returns:
            Task handle for the worker
        """
        self._task = asyncio.create_task(self.run())
        return self._task
    
    async def stop(self):
        """Request worker to stop gracefully."""
        logger.info(f"[{self.worker_id}] Stop requested")
        await self.shutdown()
    
    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._running
    
    @property
    def current_task_duration(self) -> Optional[float]:
        """Get duration of current task in seconds (None if no task)."""
        if self.current_task_start_time is None:
            return None
        return (datetime.utcnow() - self.current_task_start_time).total_seconds()
    
    def is_hung(self, timeout_seconds: float) -> bool:
        """Check if worker is hung (processing for too long)."""
        if not self._is_busy or self.current_task_start_time is None:
            return False
        duration = self.current_task_duration
        return duration is not None and duration > timeout_seconds
    
    @property
    def stats(self) -> dict:
        """Get worker statistics."""
        uptime = (
            (datetime.utcnow() - self.start_time).total_seconds()
            if self.start_time else 0
        )
        
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "running": self._running,
            "is_busy": self._is_busy,
            "current_task_duration": self.current_task_duration,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "uptime_seconds": uptime,
            "tasks_per_minute": (
                (self.tasks_processed / uptime * 60) if uptime > 0 else 0
            ),
        }
