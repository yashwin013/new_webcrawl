"""
Centralized timeout configuration.

All timeout values in one place for easy tuning and consistency across the application.
"""

from dataclasses import dataclass


@dataclass
class TimeoutConfig:
    """Centralized timeout configuration for all operations."""
    
    # Crawler timeouts
    CRAWLER_PAGE_LOAD: int = 30  # seconds to wait for page load
    CRAWLER_REQUEST: int = 30  # HTTP request timeout
    CRAWLER_NAVIGATION: int = 45  # Playwright navigation timeout
    
    # OCR timeouts
    OCR_PER_PAGE: int = 120  # seconds per page for OCR processing
    OCR_BATCH: int = 600  # 10 minutes for batch OCR
    
    # PDF processing timeouts
    PDF_PROCESSING: int = 300  # 5 minutes for PDF processing
    PDF_DOWNLOAD: int = 120  # 2 minutes to download PDF
    
    # Database timeouts (milliseconds)
    MONGODB_SERVER_SELECTION: int = 5000  # 5 seconds
    MONGODB_CONNECT: int = 10000  # 10 seconds
    MONGODB_SOCKET: int = 20000  # 20 seconds
    
    # Vector database timeouts
    QDRANT_CONNECTION: int = 30  # seconds
    QDRANT_UPSERT: int = 60  # seconds for batch upsert
    
    # Worker timeouts (for health monitoring)
    WORKER_CRAWLER: int = 300  # 5 minutes
    WORKER_PROCESSOR: int = 600  # 10 minutes
    WORKER_PDF: int = 900  # 15 minutes
    WORKER_OCR: int = 1200  # 20 minutes
    WORKER_STORAGE: int = 300  # 5 minutes
    
    # Queue timeouts
    QUEUE_GET: float = 1.0  # seconds to wait for queue item
    QUEUE_PUT: float = 5.0  # seconds to wait for queue space
    
    # Shutdown timeouts
    SHUTDOWN_GRACEFUL: float = 30.0  # seconds for graceful shutdown
    SHUTDOWN_FORCE: float = 5.0  # additional seconds before force kill


# Global singleton instance
timeout_config = TimeoutConfig()


def get_timeout_config() -> TimeoutConfig:
    """Get the global timeout configuration instance."""
    return timeout_config
