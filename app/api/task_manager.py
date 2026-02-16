"""
In-memory task manager for tracking async crawl and processing tasks.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from app.api.models import TaskStatus


@dataclass
class TaskInfo:
    """Info about a running task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    task_type: str = "crawl"  # "crawl", "batch_crawl", "process", "batch_process"
    
    # Crawl-specific
    url: Optional[str] = None
    urls: Optional[list] = None
    
    # Progress
    pages_crawled: int = 0
    total_pages: int = 0
    pages_failed: int = 0
    pdfs_downloaded: int = 0
    total_chunks: int = 0
    current_url: Optional[str] = None
    
    # Results
    urls_visited: list = field(default_factory=list)
    output_dir: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Document processing
    documents_processed: int = 0
    documents_failed: int = 0
    total_documents: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Async task handle
    _async_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _orchestrator: Optional[Any] = field(default=None, repr=False)  # Reference to orchestrator for cancellation
    
    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


class TaskManager:
    """Manages background tasks for crawling and document processing."""
    
    _instance: Optional["TaskManager"] = None
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
    
    @classmethod
    def get_instance(cls) -> "TaskManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def create_task(self, task_id: str, task_type: str = "crawl", **kwargs) -> TaskInfo:
        task = TaskInfo(task_id=task_id, task_type=task_type, **kwargs)
        self._tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs) -> Optional[TaskInfo]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        return task
    
    def list_tasks(self, task_type: Optional[str] = None) -> list[TaskInfo]:
        tasks = list(self._tasks.values())
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        
        print(f"[TaskManager] Cancelling task {task_id} (status: {task.status})")
        
        # Cancel orchestrator if present
        if hasattr(task, '_orchestrator') and task._orchestrator:
            try:
                orchestrator = task._orchestrator
                print(f"[TaskManager] Orchestrator found for task {task_id}, signaling shutdown...")
                
                # Signal orchestrator's main shutdown event
                if hasattr(orchestrator, '_shutdown_event') and orchestrator._shutdown_event:
                    orchestrator._shutdown_event.set()
                    print(f"[TaskManager] Set orchestrator shutdown event")
                
                # CRITICAL: Also shutdown the queue manager to stop workers immediately
                if hasattr(orchestrator, 'queue_manager') and orchestrator.queue_manager:
                    # Use asyncio to call async shutdown method
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        # Just set the shutdown event, don't wait for full drain
                        if orchestrator.queue_manager._shutdown_event:
                            orchestrator.queue_manager._shutdown_event.set()
                            print(f"[TaskManager] Set queue manager shutdown event - workers will stop within 1-2 seconds")
                    except Exception as e:
                        print(f"Error setting queue manager shutdown: {e}")
                
            except Exception as e:
                print(f"Error signaling orchestrator shutdown: {e}")
        else:
            print(f"[TaskManager] No orchestrator found for task {task_id}")
        
        # Cancel the async task
        if task._async_task and not task._async_task.done():
            task._async_task.cancel()
            print(f"[TaskManager] Cancelled async task for {task_id}")
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        print(f"[TaskManager] Task {task_id} marked as CANCELLED")
        return True
