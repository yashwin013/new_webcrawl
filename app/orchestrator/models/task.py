"""
Task models for the orchestrator pipeline.

Each task represents a unit of work flowing through the pipeline queues.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.crawling.models.document import Page, OCRAction


class TaskStatus(str, Enum):
    """Status of a task in the pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority for queue ordering."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class CrawlTask:
    """
    Task for crawling a website.
    
    Represents a website URL to be crawled by a crawler worker.
    """
    
    # Task identity
    task_id: str
    website_url: str
    
    # Crawl configuration
    max_pages: int = 50
    max_depth: int = 3
    crawl_session_id: str = ""
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker tracking
    worker_id: Optional[str] = None
    
    # Progress tracking
    pages_discovered: int = 0
    pages_crawled: int = 0
    pdfs_downloaded: int = 0
    
    # Error handling (reduced retries for faster failure)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 1  # Down from 3 - fail fast on problematic pages
    
    def __lt__(self, other: "CrawlTask") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def mark_started(self, worker_id: str):
        """Mark task as started by a worker."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self):
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.retry_count += 1


@dataclass
class ProcessTask:
    """
    Task for processing a crawled page (text extraction + chunking).
    
    Represents a page that needs CPU processing (no OCR yet).
    """
    
    # Task identity
    task_id: str
    page: Page
    website_url: str
    crawl_session_id: str
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker tracking
    worker_id: Optional[str] = None
    
    # Processing results
    needs_ocr: bool = False
    ocr_action: Optional[OCRAction] = None
    text_extracted: bool = False
    chunks_created: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def __lt__(self, other: "ProcessTask") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def mark_started(self, worker_id: str):
        """Mark task as started by a worker."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self, needs_ocr: bool = False, chunks: int = 0):
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.needs_ocr = needs_ocr
        self.chunks_created = chunks
        self.text_extracted = True
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.retry_count += 1


@dataclass
class OcrTask:
    """
    Task for performing OCR on a page (GPU-intensive).
    
    Represents a page that needs GPU processing via Surya OCR.
    """
    
    # Task identity
    task_id: str
    page: Page
    website_url: str
    crawl_session_id: str
    
    # OCR configuration
    ocr_action: OCRAction = OCRAction.FULL_PAGE_OCR
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker tracking
    worker_id: Optional[str] = None
    
    # OCR results
    ocr_completed: bool = False
    text_length: int = 0
    processing_time_seconds: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2  # OCR is expensive, limit retries
    
    def __lt__(self, other: "OcrTask") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def mark_started(self, worker_id: str):
        """Mark task as started by a worker."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self, text_length: int, processing_time: float):
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.ocr_completed = True
        self.text_length = text_length
        self.processing_time_seconds = processing_time
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.retry_count += 1


@dataclass
class PdfTask:
    """
    Task for processing PDF files with Docling (GPU).
    
    Represents a PDF file to be processed for text extraction.
    """
    
    # Task identity
    task_id: str
    page: Page  # Page with PDF path
    website_url: str
    crawl_session_id: str
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker tracking
    worker_id: Optional[str] = None
    
    # Processing results
    text_extracted: bool = False
    text_length: int = 0
    processing_time_seconds: float = 0.0
    chunks_created: int = 0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2  # PDF processing is expensive, limit retries
    
    def __lt__(self, other: "PdfTask") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def mark_started(self, worker_id: str):
        """Mark task as started by a worker."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self):
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.retry_count += 1


@dataclass
class StorageTask:
    """
    Task for storing processed chunks to MongoDB + Qdrant.
    
    Represents processed chunks ready for database storage.
    """
    
    # Task identity
    task_id: str
    website_url: str
    crawl_session_id: str
    
    # Data to store
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker tracking
    worker_id: Optional[str] = None
    
    # Storage results
    chunks_stored: int = 0
    vectors_stored: int = 0
    mongodb_updated: bool = False
    qdrant_updated: bool = False
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3  # Storage is critical, allow more retries
    
    def __lt__(self, other: "StorageTask") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def mark_started(self, worker_id: str):
        """Mark task as started by a worker."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
    
    def mark_completed(self, chunks_count: int, vectors_count: int):
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.chunks_stored = chunks_count
        self.vectors_stored = vectors_count
        self.mongodb_updated = True
        self.qdrant_updated = True
    
    def mark_failed(self, error: str):
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
        self.retry_count += 1
