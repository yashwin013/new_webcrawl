"""
Document models for the web scraping pipeline.

These dataclasses flow through the pipeline stages, accumulating data
as they pass through each stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from enum import Enum


class OCRAction(Enum):
    """OCR decision action types."""
    SKIP_OCR = "skip_ocr"
    OCR_IMAGES_ONLY = "ocr_images_only"
    FULL_PAGE_OCR = "full_page_ocr"


class ContentSource(Enum):
    """Source of content extraction."""
    DOM = "dom"
    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    HYBRID = "hybrid"


@dataclass
class ImageInfo:
    """Information about an image on a page."""
    width: int
    height: int
    aspect_ratio: float
    image_type: str  # "decorative", "table", "chart", "scanned_text", "unknown"
    area: int = 0
    
    def __post_init__(self):
        if self.area == 0:
            self.area = self.width * self.height


@dataclass
class PageContent:
    """Content extracted from a page."""
    text: str
    word_count: int
    char_count: int
    source: ContentSource
    images: list[ImageInfo] = field(default_factory=list)
    has_tables: bool = False
    has_charts: bool = False
    
    @classmethod
    def from_text(cls, text: str, source: ContentSource) -> "PageContent":
        """Create PageContent from raw text."""
        words = text.split()
        return cls(
            text=text,
            word_count=len(words),
            char_count=len(text),
            source=source,
        )


@dataclass
class Page:
    """A single page in the crawl (could be HTML or PDF)."""
    url: str
    depth: int = 0
    
    # File paths
    pdf_path: Optional[Path] = None
    html_path: Optional[Path] = None
    
    # Raw content
    html_content: Optional[str] = None
    dom_text: Optional[str] = None

    # Scraped images metadata
    scraped_images: list[dict] = field(default_factory=list)
    
    # Processed content (filled by stages)
    content: Optional[PageContent] = None
    ocr_action: Optional[OCRAction] = None
    ocr_reason: Optional[str] = None
    
    # Chunks (filled by ChunkerStage)
    parent_chunks: list[dict] = field(default_factory=list)
    child_chunks: list[dict] = field(default_factory=list)
    
    # Metadata
    title: Optional[str] = None
    status_code: int = 200
    content_hash: Optional[str] = None
    processing_time_ms: float = 0.0
    
    @property
    def is_pdf(self) -> bool:
        """Check if this page is a PDF."""
        return self.url.lower().endswith(".pdf") or self.pdf_path is not None
    
    @property
    def url_hash(self) -> str:
        """Generate a hash from the URL for unique identification."""
        import hashlib
        return hashlib.md5(self.url.encode()).hexdigest()[:8]
    
    @property
    def total_chunks(self) -> int:
        """Total number of chunks."""
        return len(self.parent_chunks) + len(self.child_chunks)


@dataclass
class Document:
    """
    Root document representing a crawl session.
    
    Contains all pages discovered and processed during the crawl.
    """
    start_url: str
    pages: list[Page] = field(default_factory=list)
    
    # Crawl metadata
    output_dir: Optional[Path] = None
    crawl_depth: int = 3
    max_pages: int = 50
    
    # Stats (updated as pipeline progresses)
    pages_scraped: int = 0
    pdfs_downloaded: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Total crawl duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def total_pages(self) -> int:
        """Total pages in document."""
        return len(self.pages)
    
    @property
    def total_chunks(self) -> int:
        """Total chunks across all pages."""
        return sum(p.total_chunks for p in self.pages)
    
    def add_page(self, page: Page) -> None:
        """Add a page to the document."""
        self.pages.append(page)
        # Only count as PDF download if it's a binary file without HTML content
        if page.is_pdf and not page.html_content:
            self.pdfs_downloaded += 1
        else:
            self.pages_scraped += 1
    
    def get_all_chunks(self) -> tuple[list[dict], list[dict]]:
        """Get all parent and child chunks from all pages."""
        parents = []
        children = []
        for page in self.pages:
            parents.extend(page.parent_chunks)
            children.extend(page.child_chunks)
        return parents, children
