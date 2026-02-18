"""
Configuration module for Docling project.
Loads settings from environment variables with validation.
Includes connection pooling for Qdrant and model caching.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============== Logging Configuration ==============
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("docling.log", encoding="utf-8")
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# ============== Qdrant Configuration ==============
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# ============== OpenAI Configuration ==============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "30"))

# ============== Embedding Configuration ==============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ============== GPU/Accelerator Configuration ==============
# Use GPU for document processing (OCR, etc.). Set to false to use CPU only.
# CPU mode is slower but doesn't require GPU memory and avoids OOM errors.
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
# Maximum concurrent document processing tasks (to prevent GPU OOM)
# Set to 1 to process documents sequentially, higher values for parallel processing
MAX_CONCURRENT_PROCESSING = int(os.getenv("MAX_CONCURRENT_PROCESSING", "1"))

# ============== Image Processing ==============
IMAGE_SCALE = float(os.getenv("IMAGE_SCALE", "2.0"))
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "800"))  # Max dimension in pixels
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))  # JPEG quality (1-100)
IMAGE_COMPRESS = os.getenv("IMAGE_COMPRESS", "true").lower() == "true"

# ============== OCR Preprocessing Configuration ==============
# Preprocessing level: none, basic, standard, aggressive
# - none: No preprocessing (original behavior)
# - basic: Grayscale + resize only (30-40% faster)
# - standard: + binarization + margin crop (50-60% faster)
# - aggressive: + morphological cleanup (60-70% faster)
OCR_PREPROCESSING_LEVEL = os.getenv("OCR_PREPROCESSING_LEVEL", "standard")
OCR_MAX_DIMENSION = int(os.getenv("OCR_MAX_DIMENSION", "1600"))  # Max image dimension
OCR_BINARIZATION_METHOD = os.getenv("OCR_BINARIZATION_METHOD", "otsu")  # otsu or adaptive
OCR_ENABLE_MARGIN_CROP = os.getenv("OCR_ENABLE_MARGIN_CROP", "true").lower() == "true"
OCR_ENABLE_MORPH_CLEANUP = os.getenv("OCR_ENABLE_MORPH_CLEANUP", "false").lower() == "true"
# Maximum number of detected text regions (bboxes) per page before skipping recognition
# Pages with more than this many boxes will be skipped to avoid excessive recognition work
# Default: 300 (increased from 100 to reduce skipped pages)
OCR_MAX_BBOXES_PER_PAGE = int(os.getenv("OCR_MAX_BBOXES_PER_PAGE", "300"))

# ============== PDF Processing Configuration ==============
# Skip processing for PDFs larger than this size (in MB) - just download and store
# Large PDFs can cause memory issues and long processing times
# Set to 0 to disable this feature (process all PDFs)
PDF_MAX_PROCESSING_SIZE_MB = float(os.getenv("PDF_MAX_PROCESSING_SIZE_MB", "10.0"))
# Directory for storing large PDFs that are downloaded but not processed
LARGE_PDF_STORAGE_DIR = os.getenv("LARGE_PDF_STORAGE_DIR", "data/pdfs/large_files")

# ============== OCR Decision Thresholds (Conservative - Minimize OCR Usage) ==============
# Only use OCR when absolutely necessary to save processing time
# Lower word count threshold = less aggressive, fewer PDFs sent to OCR
OCR_MIN_WORD_COUNT_SUFFICIENT = int(os.getenv("OCR_MIN_WORD_COUNT_SUFFICIENT", "50"))  # PDFs with ≤50 words go to OCR backlog
OCR_SCANNED_PDF_MAX_WORDS = int(os.getenv("OCR_SCANNED_PDF_MAX_WORDS", "10"))  # Lowered from 20 to 10
# Image thresholds - only OCR large, text-bearing images
OCR_MIN_TEXT_BEARING_IMAGES = int(os.getenv("OCR_MIN_TEXT_BEARING_IMAGES", "5"))  # Changed from 3 to 5
OCR_MIN_TEXT_BEARING_RATIO = float(os.getenv("OCR_MIN_TEXT_BEARING_RATIO", "0.7"))  # Changed from 0.5 to 0.7
OCR_MIN_TEXT_BEARING_AREA = int(os.getenv("OCR_MIN_TEXT_BEARING_AREA", "400000"))  # Changed from 200k to 400k (larger images only)
OCR_DECORATIVE_MAX_SIZE = int(os.getenv("OCR_DECORATIVE_MAX_SIZE", "500"))  # Changed from 300 to 500 (skip small images)

# ============== Chunking Configuration ==============
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))  # 15% overlap (NVIDIA recommended)
CHUNK_MERGE_PEERS = os.getenv("CHUNK_MERGE_PEERS", "true").lower() == "true"
CHUNK_INCLUDE_METADATA = os.getenv("CHUNK_INCLUDE_METADATA", "true").lower() == "true"
CHUNK_HEADING_AS_METADATA = os.getenv("CHUNK_HEADING_AS_METADATA", "true").lower() == "true"
CHUNK_ENABLE_OVERLAP = os.getenv("CHUNK_ENABLE_OVERLAP", "true").lower() == "true"
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "hybrid")  # Options: hybrid, hierarchical, page, semantic
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "20"))  # Merge chunks smaller than this
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.5"))  # For semantic chunking

# ============== Deduplication Configuration ==============
DEDUP_ENABLED = os.getenv("DEDUP_ENABLED", "true").lower() == "true"
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.92"))  # 92% similarity

# ============== MongoDB Configuration ==============
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "crawl")
PDF_STORAGE_PATH = Path(os.getenv("PDF_STORAGE_PATH", str(Path(__file__).parent.parent / "data" / "pdfs")))

# ============== Paths ==============
BASE_DIR = Path(__file__).parent.parent  # Points to web_crawl root
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
exported_images = os.getenv("EXPORTED_IMAGES_DIR", str(OUTPUT_DIR / "images"))


# ============== Validation ==============
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_qdrant_config() -> None:
    """Validate that Qdrant configuration is properly set."""
    if not QDRANT_URL:
        raise ConfigurationError(
            "QDRANT_URL is not set. Please set it in your .env file."
        )
    if not QDRANT_API_KEY:
        raise ConfigurationError(
            "QDRANT_API_KEY is not set. Please set it in your .env file."
        )


def validate_pdf_file(pdf_path: Path) -> None:
    """
    Validate that a PDF file exists and is valid.
    
    Args:
        pdf_path: Path to the PDF file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PDF or is too large/small
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    # Check file size (min 1KB, max 100MB)
    file_size = pdf_path.stat().st_size
    if file_size < 1024:
        raise ValueError(f"PDF file is too small ({file_size} bytes): {pdf_path}")
    if file_size > 100 * 1024 * 1024:
        raise ValueError(f"PDF file is too large ({file_size / 1024 / 1024:.1f}MB): {pdf_path}")


def ensure_output_dirs() -> None:
    """Ensure output directories exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    Path(exported_images).mkdir(parents=True, exist_ok=True)


# ============== Connection Pooling (Singleton Pattern) ==============
_qdrant_client: Optional["QdrantClient"] = None
_embedding_model: Optional["SentenceTransformer"] = None
_embedding_vector_size: Optional[int] = None

_logger = get_logger("config")


def get_qdrant_client():
    """
    Get or create a singleton Qdrant client.
    Reuses the same connection across the application.
    """
    global _qdrant_client
    
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        
        validate_qdrant_config()
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,  # Connection timeout
        )
        # Test connection
        _qdrant_client.get_collections()
        _logger.info(f"Created Qdrant client connection to {QDRANT_URL}")
    
    return _qdrant_client


def get_embedding_model():
    """
    Get or create a singleton embedding model.
    Reuses the same model across the application for efficiency.
    
    Returns:
        Tuple of (model, vector_size)
    """
    global _embedding_model, _embedding_vector_size
    
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        _embedding_vector_size = _embedding_model.get_sentence_embedding_dimension()
        _logger.info(f"Loaded embedding model '{EMBEDDING_MODEL}' on {device.upper()} (dim={_embedding_vector_size})")
    
    return _embedding_model, _embedding_vector_size


# ============== Tokenizer for Accurate Token Counting ==============
_tokenizer = None


def get_tokenizer():
    """
    Get or create a singleton HuggingFace tokenizer.
    Uses the same model as embeddings for consistent token counting.
    
    Returns:
        AutoTokenizer instance
    """
    global _tokenizer
    
    if _tokenizer is None:
        from transformers import AutoTokenizer
        
        # Use the same model as embeddings for consistency
        tokenizer_model = f"sentence-transformers/{EMBEDDING_MODEL}"
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        _logger.info(f"Loaded tokenizer for '{tokenizer_model}'")
    
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens in text using HuggingFace tokenizer.
    
    This is more accurate than word counting (text.split()) because:
    - Subword tokenization: "Empanelment" → multiple tokens
    - Special characters: URLs, abbreviations split into many tokens
    - Matches what the embedding model actually sees
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def close_connections() -> None:
    """Close all singleton connections. Call on application shutdown."""
    global _qdrant_client, _embedding_model
    
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None
        _logger.info("Closed Qdrant client connection")
    
    _embedding_model = None


# ============== Settings Class (for compatibility with processdocling) ==============
class AppConfig:
    """Configuration class for backward compatibility."""
    QDRANT_URL = QDRANT_URL
    QDRANT_API_KEY = QDRANT_API_KEY
    QDRANT_COLLECTION = QDRANT_COLLECTION
    EMBEDDING_MODEL = EMBEDDING_MODEL
    EMBEDDING_BATCH_SIZE = EMBEDDING_BATCH_SIZE
    OPENAI_API_KEY = OPENAI_API_KEY
    OPENAI_MODEL = OPENAI_MODEL
    RAG_TOP_K = RAG_TOP_K
    UPLOAD_DIR = UPLOAD_DIR
    exported_images = exported_images
    BASE_DIR = BASE_DIR
    OUTPUT_DIR = OUTPUT_DIR
    DATA_DIR = DATA_DIR
    CHUNK_MAX_TOKENS = CHUNK_MAX_TOKENS
    CHUNK_OVERLAP_TOKENS = CHUNK_OVERLAP_TOKENS
    CHUNK_MIN_TOKENS = CHUNK_MIN_TOKENS
    OCR_MAX_BBOXES_PER_PAGE = OCR_MAX_BBOXES_PER_PAGE
    MONGODB_URL = MONGODB_URL
    MONGODB_DATABASE = MONGODB_DATABASE


# Singleton instance for backward compatibility
app_config = AppConfig()

# Also expose as 'settings' for new code
settings = app_config
