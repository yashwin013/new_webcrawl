"""
Document schema models for MongoDB storage.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class DocumentStatus(str, Enum):
    """Processing status for documents."""
    PENDING = "pending"           # Queued for chunking/vectorization
    PROCESSING = "processing"     # Currently being processed
    VECTORIZED = "vectorized"     # Successfully chunked and stored in Qdrant
    FAILED = "failed"             # Processing failed
    STORED = "stored"             # Stored only, no chunking needed (direct PDF downloads)


class PageDocument(BaseModel):
    """
    A document (HTML page or PDF) nested within a visited URL.
    Supports both web pages and PDF files in the nested structure.
    """
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_file: str = Field(..., description="Original filename or URL")
    source_url: str = Field(..., description="URL where document was crawled from")
    file_path: str = Field(default="", description="Storage path on filesystem (empty for HTML pages)")
    document_type: str = Field(default="html", description="Type: 'html' or 'pdf'")
    
    # Processing state
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    vector_count: int = Field(default=0, description="Number of chunks stored in Qdrant")
    
    # File metadata
    file_size: int = Field(default=0, description="File size in bytes")
    page_count: int = Field(default=0, description="Number of pages")
    mime_type: str = Field(default="text/html")
    
    # Crawl context
    crawl_session_id: str = Field(..., description="Groups PDFs by crawl run")
    crawl_depth: int = Field(default=0, description="Depth in crawl tree")
    
    # Soft delete
    is_deleted: bool = Field(default=False)
    
    # Pipeline tracking flags (0/1 strings for compatibility)
    is_crawled: str = Field(default="0", description="0=not crawled, 1=crawled successfully")
    is_vectorized: str = Field(default="0", description="0=not vectorized, 1=vectorized")
    is_ocr_required: str = Field(default="0", description="0=no OCR needed, 1=needs OCR")
    is_ocr_completed: str = Field(default="0", description="0=OCR pending, 1=OCR done")
    
    # Processing statistics
    total_pages: int = Field(default=0, description="Total pages found during crawl")
    pages_with_text: int = Field(default=0, description="Pages with extractable text")
    pages_needing_ocr: int = Field(default=0, description="Pages requiring OCR")
    
    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
    
    def to_mongo_dict(self) -> dict:
        """Convert to MongoDB-compatible dictionary."""
        data = self.model_dump()
        data["fileId"] = data.pop("file_id")
        data["originalFile"] = data.pop("original_file")
        data["sourceUrl"] = data.pop("source_url")
        data["filePath"] = data.pop("file_path")
        data["documentType"] = data.pop("document_type")
        data["vectorCount"] = data.pop("vector_count")
        data["fileSize"] = data.pop("file_size")
        data["pageCount"] = data.pop("page_count")
        data["mimeType"] = data.pop("mime_type")
        data["crawlSessionId"] = data.pop("crawl_session_id")
        data["crawlDepth"] = data.pop("crawl_depth")
        data["isDeleted"] = data.pop("is_deleted")
        data["isCrawled"] = data.pop("is_crawled")
        data["isVectorized"] = data.pop("is_vectorized")
        data["isOcrRequired"] = data.pop("is_ocr_required")
        data["isOcrCompleted"] = data.pop("is_ocr_completed")
        data["totalPages"] = data.pop("total_pages")
        data["pagesWithText"] = data.pop("pages_with_text")
        data["pagesNeedingOcr"] = data.pop("pages_needing_ocr")
        data["createdAt"] = data.pop("created_at")
        data["updatedAt"] = data.pop("updated_at")
        return data
    
    @classmethod
    def from_mongo_dict(cls, data: dict) -> "PageDocument":
        """Create from MongoDB document."""
        return cls(
            file_id=data.get("fileId", ""),
            original_file=data.get("originalFile", ""),
            source_url=data.get("sourceUrl", ""),
            file_path=data.get("filePath", ""),
            document_type=data.get("documentType", "html"),
            status=data.get("status", DocumentStatus.PENDING),
            vector_count=data.get("vectorCount", 0),
            file_size=data.get("fileSize", 0),
            page_count=data.get("pageCount", 0),
            mime_type=data.get("mimeType", "text/html"),
            crawl_session_id=data.get("crawlSessionId", ""),
            crawl_depth=data.get("crawlDepth", 0),
            is_deleted=data.get("isDeleted", False),
            is_crawled=data.get("isCrawled", "0"),
            is_vectorized=data.get("isVectorized", "0"),
            is_ocr_required=data.get("isOcrRequired", "0"),
            is_ocr_completed=data.get("isOcrCompleted", "0"),
            total_pages=data.get("totalPages", 0),
            pages_with_text=data.get("pagesWithText", 0),
            pages_needing_ocr=data.get("pagesNeedingOcr", 0),
            created_at=data.get("createdAt", datetime.utcnow()),
            updated_at=data.get("updatedAt", datetime.utcnow()),
        )


# Backward compatibility alias
PdfDocument = PageDocument


class VisitedUrl(BaseModel):
    """
    A URL visited during crawling, containing all documents (HTML pages and PDFs) found.
    """
    url: str = Field(..., description="The visited URL")
    crawl_depth: int = Field(default=0, description="Depth in crawl tree")
    visited_at: datetime = Field(default_factory=datetime.utcnow)
    documents: List[PageDocument] = Field(default_factory=list, description="Documents (HTML/PDF) at this URL")
    
    # Backward compatibility
    @property
    def pdfs(self) -> List[PageDocument]:
        """Alias for backward compatibility."""
        return self.documents
    
    def to_mongo_dict(self) -> dict:
        """Convert to MongoDB-compatible dictionary."""
        return {
            "url": self.url,
            "crawlDepth": self.crawl_depth,
            "visitedAt": self.visited_at,
            "documents": [doc.to_mongo_dict() for doc in self.documents]
        }
    
    @classmethod
    def from_mongo_dict(cls, data: dict) -> "VisitedUrl":
        """Create from MongoDB document."""
        # Support both old 'pdfs' and new 'documents' fields
        docs_data = data.get("documents", data.get("pdfs", []))
        return cls(
            url=data.get("url", ""),
            crawl_depth=data.get("crawlDepth", 0),
            visited_at=data.get("visitedAt", datetime.utcnow()),
            documents=[PageDocument.from_mongo_dict(doc) for doc in docs_data]
        )


class WebsiteCrawl(BaseModel):
    """
    Top-level document representing a crawled website.
    Contains all visited URLs and their documents nested within.
    """
    website_url: str = Field(..., description="Base URL of the website")
    crawl_session_id: str = Field(..., description="Unique session ID for this crawl")
    visited_urls: List[VisitedUrl] = Field(default_factory=list, description="All URLs visited on this website")
    
    # Website-level tracking flags
    is_visited: str = Field(default="0", description="0=not all URLs visited, 1=all URLs visited")
    is_crawled: str = Field(default="0", description="0=not all documents crawled, 1=all documents crawled")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_mongo_dict(self) -> dict:
        """Convert to MongoDB-compatible dictionary."""
        return {
            "websiteUrl": self.website_url,
            "crawlSessionId": self.crawl_session_id,
            "visitedUrls": [url.to_mongo_dict() for url in self.visited_urls],
            "isVisited": self.is_visited,
            "isCrawled": self.is_crawled,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at
        }
    
    @classmethod
    def from_mongo_dict(cls, data: dict) -> "WebsiteCrawl":
        """Create from MongoDB document."""
        return cls(
            website_url=data.get("websiteUrl", ""),
            crawl_session_id=data.get("crawlSessionId", ""),
            visited_urls=[VisitedUrl.from_mongo_dict(url) for url in data.get("visitedUrls", [])],
            is_visited=data.get("isVisited", "0"),
            is_crawled=data.get("isCrawled", "0"),
            created_at=data.get("createdAt", datetime.utcnow()),
            updated_at=data.get("updatedAt", datetime.utcnow())
        )
    
    def check_crawl_status(self) -> tuple[str, str]:
        """Check if all URLs are visited and all documents are crawled.
        
        Returns:
            Tuple of (is_visited, is_crawled) as "0" or "1"
        """
        if not self.visited_urls:
            return "0", "0"
        
        # Check if all documents in all visited URLs are crawled
        all_crawled = True
        has_documents = False
        
        for visited_url in self.visited_urls:
            for doc in visited_url.documents:
                has_documents = True
                if doc.is_crawled != "1":
                    all_crawled = False
                    break
            if not all_crawled:
                break
        
        # is_visited is "1" if we have at least one visited URL
        is_visited = "1" if len(self.visited_urls) > 0 else "0"
        
        # is_crawled is "1" only if we have documents AND all are crawled
        is_crawled = "1" if (has_documents and all_crawled) else "0"
        
        return is_visited, is_crawled


class CrawledDocument(BaseModel):
    """
    Schema for crawled PDF documents stored in MongoDB.
    
    Tracks document metadata, processing status, and audit trail.
    """
    file_id: str = Field(default_factory=lambda: f"{uuid.uuid4()}.pdf")
    original_file: str = Field(..., description="Original filename")
    source_url: str = Field(..., description="URL where PDF was crawled from")
    file_path: str = Field(..., description="Storage path on filesystem")
    
    # Processing state
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    vector_count: int = Field(default=0, description="Number of chunks stored in Qdrant")
    error_message: Optional[str] = Field(default=None, description="Error details if failed")
    
    # File metadata
    file_size: int = Field(default=0, description="File size in bytes")
    page_count: int = Field(default=0, description="Number of pages")
    mime_type: str = Field(default="application/pdf")
    
    # Crawl context
    crawl_session_id: str = Field(..., description="Groups PDFs by crawl run")
    crawl_depth: int = Field(default=0, description="Depth in crawl tree")
    
    # Soft delete
    is_deleted: bool = Field(default=False)
    
    # Pipeline tracking flags (0/1 strings for compatibility)
    is_crawled: str = Field(default="0", description="0=not crawled, 1=crawled successfully")
    is_vectorized: str = Field(default="0", description="0=not vectorized, 1=vectorized")
    is_ocr_required: str = Field(default="0", description="0=no OCR needed, 1=needs OCR")
    is_ocr_completed: str = Field(default="0", description="0=OCR pending, 1=OCR done")
    
    # Stage timestamps
    crawl_started_at: Optional[datetime] = Field(default=None)
    crawl_completed_at: Optional[datetime] = Field(default=None)
    ocr_started_at: Optional[datetime] = Field(default=None)
    ocr_completed_at: Optional[datetime] = Field(default=None)
    vectorization_started_at: Optional[datetime] = Field(default=None)
    vectorization_completed_at: Optional[datetime] = Field(default=None)
    
    # Processing statistics
    total_pages: int = Field(default=0, description="Total pages found during crawl")
    pages_with_text: int = Field(default=0, description="Pages with extractable text")
    pages_needing_ocr: int = Field(default=0, description="Pages requiring OCR")
    
    # Worker tracking (for debugging)
    crawled_by_worker: Optional[str] = Field(default=None)
    processed_by_worker: Optional[str] = Field(default=None)
    
    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="crawler")
    updated_by: str = Field(default="crawler")
    
    class Config:
        use_enum_values = True
    
    def to_mongo_dict(self) -> dict:
        """Convert to MongoDB-compatible dictionary."""
        data = self.model_dump()
        # Rename file_id to fileId for MongoDB convention
        data["fileId"] = data.pop("file_id")
        data["originalFile"] = data.pop("original_file")
        data["sourceUrl"] = data.pop("source_url")
        data["filePath"] = data.pop("file_path")
        data["vectorCount"] = data.pop("vector_count")
        data["errorMessage"] = data.pop("error_message")
        data["fileSize"] = data.pop("file_size")
        data["pageCount"] = data.pop("page_count")
        data["mimeType"] = data.pop("mime_type")
        data["crawlSessionId"] = data.pop("crawl_session_id")
        data["crawlDepth"] = data.pop("crawl_depth")
        data["isDeleted"] = data.pop("is_deleted")
        # Pipeline tracking
        data["isCrawled"] = data.pop("is_crawled")
        data["isVectorized"] = data.pop("is_vectorized")
        data["isOcrRequired"] = data.pop("is_ocr_required")
        data["isOcrCompleted"] = data.pop("is_ocr_completed")
        # Timestamps
        data["crawlStartedAt"] = data.pop("crawl_started_at")
        data["crawlCompletedAt"] = data.pop("crawl_completed_at")
        data["ocrStartedAt"] = data.pop("ocr_started_at")
        data["ocrCompletedAt"] = data.pop("ocr_completed_at")
        data["vectorizationStartedAt"] = data.pop("vectorization_started_at")
        data["vectorizationCompletedAt"] = data.pop("vectorization_completed_at")
        # Statistics
        data["totalPages"] = data.pop("total_pages")
        data["pagesWithText"] = data.pop("pages_with_text")
        data["pagesNeedingOcr"] = data.pop("pages_needing_ocr")
        # Worker tracking
        data["crawledByWorker"] = data.pop("crawled_by_worker")
        data["processedByWorker"] = data.pop("processed_by_worker")
        # Audit
        data["createdAt"] = data.pop("created_at")
        data["updatedAt"] = data.pop("updated_at")
        data["createdBy"] = data.pop("created_by")
        data["updatedBy"] = data.pop("updated_by")
        return data
    
    @classmethod
    def from_mongo_dict(cls, data: dict) -> "CrawledDocument":
        """Create from MongoDB document."""
        return cls(
            file_id=data.get("fileId", ""),
            original_file=data.get("originalFile", ""),
            source_url=data.get("sourceUrl", ""),
            file_path=data.get("filePath", ""),
            status=data.get("status", DocumentStatus.PENDING),
            vector_count=data.get("vectorCount", 0),
            error_message=data.get("errorMessage"),
            file_size=data.get("fileSize", 0),
            page_count=data.get("pageCount", 0),
            mime_type=data.get("mimeType", "application/pdf"),
            crawl_session_id=data.get("crawlSessionId", ""),
            crawl_depth=data.get("crawlDepth", 0),
            is_deleted=data.get("isDeleted", False),
            # Pipeline tracking
            is_crawled=data.get("isCrawled", "0"),
            is_vectorized=data.get("isVectorized", "0"),
            is_ocr_required=data.get("isOcrRequired", "0"),
            is_ocr_completed=data.get("isOcrCompleted", "0"),
            # Timestamps
            crawl_started_at=data.get("crawlStartedAt"),
            crawl_completed_at=data.get("crawlCompletedAt"),
            ocr_started_at=data.get("ocrStartedAt"),
            ocr_completed_at=data.get("ocrCompletedAt"),
            vectorization_started_at=data.get("vectorizationStartedAt"),
            vectorization_completed_at=data.get("vectorizationCompletedAt"),
            # Statistics
            total_pages=data.get("totalPages", 0),
            pages_with_text=data.get("pagesWithText", 0),
            pages_needing_ocr=data.get("pagesNeedingOcr", 0),
            # Worker tracking
            crawled_by_worker=data.get("crawledByWorker"),
            processed_by_worker=data.get("processedByWorker"),
            # Audit
            created_at=data.get("createdAt", datetime.utcnow()),
            updated_at=data.get("updatedAt", datetime.utcnow()),
            created_by=data.get("createdBy", "crawler"),
            updated_by=data.get("updatedBy", "crawler"),
        )
