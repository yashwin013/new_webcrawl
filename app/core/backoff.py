"""
Exponential backoff utilities with jitter for retry logic.

Provides exponential backoff with configurable parameters to handle
transient failures more effectively than fixed delays.
"""

import asyncio
import random
from typing import Optional, Callable, Any
from functools import wraps

from app.config import get_logger

logger = get_logger(__name__)


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter.
    
    Usage:
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)
        for attempt in range(max_attempts):
            try:
                result = await operation()
                break
            except TransientError:
                await backoff.wait(attempt)
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize backoff calculator.
        
        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Exponential multiplier (default 2.0 = double each time)
            jitter: Add random jitter to prevent thundering herd
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = min(self.base_delay * (self.multiplier ** attempt), self.max_delay)
        
        # Add jitter (±25% random variance)
        if self.jitter:
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)
    
    async def wait(self, attempt: int) -> None:
        """
        Wait for the calculated backoff delay.
        
        Args:
            attempt: Attempt number (0-indexed)
        """
        delay = self.calculate_delay(attempt)
        if delay > 0:
            logger.debug(f"Backing off for {delay:.2f}s (attempt {attempt + 1})")
            await asyncio.sleep(delay)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to retry async functions with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_attempts=3, base_delay=1.0, exceptions=(TimeoutError,))
        async def fetch_data():
            ...
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function(exception, attempt)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            backoff = ExponentialBackoff(base_delay=base_delay, max_delay=max_delay)
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt)
                        
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        await backoff.wait(attempt)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )
            
            # All attempts exhausted
            raise last_exception
        
        return wrapper
    return decorator


class LinearBackoff:
    """
    Linear backoff calculator (fixed incremental delays).
    
    Simpler than exponential, useful for operations that shouldn't
    grow delays too quickly.
    """
    
    def __init__(self, increment: float = 1.0, max_delay: float = 10.0):
        """
        Initialize linear backoff.
        
        Args:
            increment: Delay increment per attempt (seconds)
            max_delay: Maximum delay (seconds)
        """
        self.increment = increment
        self.max_delay = max_delay
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        return min(self.increment * (attempt + 1), self.max_delay)
    
    async def wait(self, attempt: int) -> None:
        """Wait for linear backoff delay."""
        delay = self.calculate_delay(attempt)
        if delay > 0:
            await asyncio.sleep(delay)
