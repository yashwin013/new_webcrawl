"""
ThreadPool Executor Manager with reference counting.

Manages shared thread pool executors with proper lifecycle and cleanup.
Ensures executors are only shutdown when all references are released.
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger("executor_manager")


class ExecutorManager:
    """
    Manages ThreadPoolExecutor with reference counting for safe shutdown.
    
    Features:
    - Shared executor across multiple components
    - Reference counting to prevent premature shutdown
    - Automatic cleanup when no references remain
    - Thread-safe operations
    """
    
    _instance: Optional["ExecutorManager"] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(self, max_workers: int = 4):
        """Initialize executor manager (use get_instance() instead)."""
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers = max_workers
        self._reference_count = 0
        self._ref_lock = asyncio.Lock()
        
    @classmethod
    def get_instance(cls, max_workers: int = 4) -> "ExecutorManager":
        """Get singleton instance of executor manager."""
        if cls._instance is None:
            cls._instance = cls(max_workers)
        return cls._instance
    
    async def acquire(self) -> ThreadPoolExecutor:
        """
        Acquire a reference to the executor.
        Creates executor on first acquisition.
        
        Returns:
            ThreadPoolExecutor instance
        """
        async with self._ref_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="docling_worker"
                )
                logger.info(f"✓ Created ThreadPoolExecutor with {self._max_workers} workers")
            
            self._reference_count += 1
            logger.debug(f"Executor acquired (refs: {self._reference_count})")
            return self._executor
    
    async def release(self) -> None:
        """
        Release a reference to the executor.
        Shuts down executor when reference count reaches zero.
        """
        async with self._ref_lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.debug(f"Executor released (refs: {self._reference_count})")
                
                # Shutdown when no more references
                if self._reference_count == 0 and self._executor is not None:
                    try:
                        logger.info("Shutting down ThreadPoolExecutor (no references remain)...")
                        self._executor.shutdown(wait=True, cancel_futures=False)
                        self._executor = None
                        logger.info("✓ ThreadPoolExecutor shutdown complete")
                    except Exception as e:
                        logger.error(f"Error shutting down executor: {e}", exc_info=True)
    
    @asynccontextmanager
    async def executor_context(self):
        """
        Context manager for safe executor usage.
        
        Usage:
            async with executor_manager.executor_context() as executor:
                result = await loop.run_in_executor(executor, blocking_func)
        """
        executor = await self.acquire()
        try:
            yield executor
        finally:
            await self.release()
    
    async def force_shutdown(self) -> None:
        """
        Force shutdown of executor regardless of reference count.
        Use only on application shutdown.
        """
        async with self._ref_lock:
            if self._executor is not None:
                try:
                    if self._reference_count > 0:
                        logger.info(f"Force shutdown executor (active refs: {self._reference_count})")
                    else:
                        logger.info("Shutting down executor...")
                    
                    self._executor.shutdown(wait=True, cancel_futures=True)
                    self._executor = None
                    self._reference_count = 0
                    logger.info("✓ Executor force shutdown complete")
                except Exception as e:
                    logger.error(f"Error during force shutdown: {e}", exc_info=True)
    
    @property
    def is_active(self) -> bool:
        """Check if executor is currently active."""
        return self._executor is not None
    
    @property
    def reference_count(self) -> int:
        """Get current reference count."""
        return self._reference_count


# Global singleton instance
executor_manager = ExecutorManager.get_instance(max_workers=8)  # Increased for heavy PDF processing


# Application lifecycle hooks
async def startup_executor():
    """Call this on application startup if you want to pre-create executor."""
    await executor_manager.acquire()


async def shutdown_executor():
    """Call this on application shutdown to force cleanup."""
    await executor_manager.force_shutdown()
