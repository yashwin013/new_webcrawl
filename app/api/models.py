"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
import uuid


# ======================== Enums ========================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    VECTORIZED = "vectorized"
    FAILED = "failed"
    STORED = "stored"


# ======================== Crawl Models ========================

class CrawlStartRequest(BaseModel):
    """Request body for starting a single URL crawl."""
    url: str = Field(..., description="URL to start crawling from")
    max_pages: int = Field(default=50, ge=1, le=1000, description="Maximum pages to crawl")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum crawl depth")
    crawl_session_id: Optional[str] = Field(
        default=None,
        description="Custom session ID (auto-generated if not provided)"
    )


class CrawlBatchRequest(BaseModel):
    """Request body for starting a batch crawl of multiple URLs."""
    urls: List[str] = Field(..., min_length=1, max_length=50, description="List of URLs to crawl")
    max_pages: int = Field(default=50, ge=1, le=1000, description="Maximum pages per site")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum crawl depth per site")


class CrawlTaskResponse(BaseModel):
    """Response for a crawl task creation."""
    task_id: str = Field(..., description="Task ID for tracking")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    message: str = Field(default="Crawl task created successfully")
    url: Optional[str] = Field(default=None, description="URL being crawled (single crawl)")
    urls: Optional[List[str]] = Field(default=None, description="URLs being crawled (batch)")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CrawlStatusResponse(BaseModel):
    """Response for crawl status check."""
    task_id: str
    status: TaskStatus
    progress: Dict[str, Any] = Field(default_factory=dict)
    pages_crawled: int = 0
    total_pages: int = 0
    pages_failed: int = 0
    current_url: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None


class CrawlResultsResponse(BaseModel):
    """Response for crawl results."""
    task_id: str
    status: TaskStatus
    urls_visited: List[str] = Field(default_factory=list)
    documents_extracted: int = 0
    total_chunks: int = 0
    pdfs_downloaded: int = 0
    pages_scraped: int = 0
    pages_failed: int = 0
    duration_seconds: float = 0.0
    output_dir: Optional[str] = None


class StopCrawlResponse(BaseModel):
    """Response for stopping a crawl."""
    task_id: str
    status: TaskStatus
    message: str


# ======================== Document Models ========================

class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str = Field(..., description="Unique document ID")
    file_name: str
    file_size: int
    status: DocumentStatusEnum = DocumentStatusEnum.PENDING
    message: str = "Document uploaded successfully"


class DocumentProcessRequest(BaseModel):
    """Request for processing a document."""
    file_id: str = Field(..., description="File ID of the document to process")
    file_path: Optional[str] = Field(default=None, description="File path (if not in default location)")
    store_vectors: bool = Field(default=True, description="Whether to store vectors in Qdrant")


class DocumentBatchProcessRequest(BaseModel):
    """Request for batch processing documents."""
    file_ids: Optional[List[str]] = Field(default=None, description="List of file IDs to process")
    folder_path: Optional[str] = Field(default=None, description="Folder containing documents to process")
    store_vectors: bool = Field(default=True, description="Whether to store vectors in Qdrant")


class DocumentProcessResponse(BaseModel):
    """Response for document processing."""
    task_id: str
    file_id: Optional[str] = None
    status: str
    message: str
    chunks_created: int = 0
    vector_count: int = 0
    processing_time_seconds: float = 0.0


class DocumentBatchProcessResponse(BaseModel):
    """Response for batch document processing."""
    task_id: str
    status: str
    message: str
    total_documents: int = 0
    processed: int = 0
    failed: int = 0
    total_chunks: int = 0


class DocumentDetailResponse(BaseModel):
    """Response for document details."""
    file_id: str
    original_file: str
    source_url: str
    file_path: str
    document_type: str = "pdf"
    status: str
    vector_count: int = 0
    file_size: int = 0
    page_count: int = 0
    mime_type: str = "application/pdf"
    crawl_session_id: str = ""
    is_deleted: bool = False
    is_crawled: str = "0"
    is_vectorized: str = "0"
    is_ocr_required: str = "0"
    is_ocr_completed: str = "0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentStatusResponse(BaseModel):
    """Response for document processing status."""
    file_id: str
    status: str
    vector_count: int = 0
    page_count: int = 0
    is_vectorized: str = "0"
    is_ocr_required: str = "0"
    is_ocr_completed: str = "0"
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[DocumentDetailResponse]
    total: int
    limit: int
    offset: int


class DeleteDocumentResponse(BaseModel):
    """Response for deleting a document."""
    file_id: str
    deleted: bool
    message: str


# ======================== General Models ========================

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
