"""
Configuration models for the web scraping pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CrawlConfig:
    """Configuration for the crawler stage."""
    max_depth: int = 3
    max_pages: int = 50
    request_delay: float = 1.0
    max_delay: float = 30.0
    max_concurrent: int = 5
    
    # Feature flags
    use_sitemap: bool = True
    respect_robots: bool = False  # Disabled by default - most docs sites want to be indexed
    follow_external_links: bool = False
    
    # Content filtering
    skip_login_pages: bool = True
    skip_404: bool = True
    skip_duplicates: bool = True
    
    # URL patterns
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    
    # Timeouts
    page_timeout_ms: int = 30000
    idle_timeout_ms: int = 10000
    
    # State persistence
    state_file: Optional[Path] = None
    save_state_interval: int = 10


@dataclass
class TextExtractorConfig:
    """Configuration for text extraction stage."""
    min_chars_for_skip_ocr: int = 600
    min_words_for_skip_ocr: int = 70
    min_unique_words: int = 100


@dataclass
class OCRConfig:
    """Configuration for OCR stages."""
    # Thresholds for OCR triggering
    min_text_bearing_images: int = 1  # Aggressive: triggers if even 1 valid image
    min_text_bearing_ratio: float = 0.5
    min_word_count_sufficient: int = 100
    scanned_pdf_max_words: int = 50
    
    # Image classification
    min_text_bearing_area: int = 50000  # ~225x225 pixels
    decorative_max_size: int = 200  # Images below this are decorative
    
    # GPU settings
    use_gpu: bool = True
    batch_size: int = 4


@dataclass
class ChunkerConfig:
    """Configuration for the chunker stage."""
    min_tokens: int = 50
    max_tokens: int = 500
    chunk_overlap_words: int = 30
    
    # DOM chunking
    dom_chunk_min_words: int = 100
    dom_chunk_max_words: int = 300


@dataclass
class PipelineConfig:
    """
    Master configuration for the entire pipeline.
    
    Contains sub-configs for each stage.
    """
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("outputs/scraped"))
    
    # Stage configs
    crawl: CrawlConfig = field(default_factory=CrawlConfig)
    text_extractor: TextExtractorConfig = field(default_factory=TextExtractorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    
    # Pipeline behavior
    skip_existing: bool = True
    save_intermediate: bool = True  # Save results after each stage
    parallel_processing: bool = False
    max_workers: int = 4
    
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Ensure output_dir is a Path."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
