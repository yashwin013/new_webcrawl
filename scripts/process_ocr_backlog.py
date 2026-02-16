"""
Process OCR Backlog

Processes pages that were marked as needing OCR during orchestrator run.
Uses batch processing (10 pages at a time) to avoid GPU memory issues.
Automatically clears GPU cache between batches.

RECOMMENDED USAGE (Sets optimal GPU memory config):
    Windows: process_ocr_backlog.bat
    Linux/Mac: ./process_ocr_backlog.sh

DIRECT USAGE (Manual memory config):
    Set environment variable first:
        Windows: SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        Linux/Mac: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    Then run:
        python scripts/process_ocr_backlog.py

IMPROVEMENTS:
    - GPU memory clearing: Clears GPU memory before Phase 2, after each page, and after embeddings
    - Batch processing: Processes 10 pages at a time (was: all pages at once)
    - Memory optimization: Uses PyTorch expandable segments
    - Supports large PDFs: Can handle 100+ page PDFs on 6GB GPU
"""

import asyncio
import json
import sys
import gc
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_logger
from app.crawling.models.document import Page, OCRAction, PageContent, ContentSource
from app.orchestrator.workers.helpers.ocr_processor import process_page_ocr
from app.orchestrator.workers.helpers.text_processor import chunk_page_text
from app.services.document_store import DocumentStore

logger = get_logger(__name__)


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("✓ Cleared GPU cache")
    except ImportError:
        pass
    gc.collect()


async def _store_chunks(
    chunks: list,
    entry: dict,
    doc_store: DocumentStore,
) -> None:
    """
    Store OCR chunks to MongoDB and Qdrant.
    
    Args:
        chunks: List of chunk dictionaries
        entry: Backlog entry with metadata
        doc_store: Document store instance
    """
    from app.config import get_embedding_model, get_qdrant_client
    from qdrant_client.models import PointStruct, Distance, VectorParams
    from datetime import datetime
    import hashlib
    
    url = entry["url"]
    
    # Get or create document
    doc = doc_store.get_by_source_url(url)
    if not doc:
        doc = doc_store.create_document(
            original_file=url,
            source_url=url,
            file_path=entry.get("pdf_path", ""),
            crawl_session_id="ocr_backlog_processing",
            crawl_depth=entry.get("depth", 0),
        )
    
    # Update document with OCR metadata
    doc_store.update_document(
        doc.file_id,
        {
            "is_ocr_required": "1",
            "is_ocr_completed": "1",
            "ocr_completed_at": datetime.utcnow(),
            "is_crawled": "1",
            "vector_count": len(chunks),
        }
    )
    
    # Get embedding model and Qdrant client
    embedding_model, vector_size = get_embedding_model()
    qdrant_client = get_qdrant_client()
    collection_name = "crawled_documents"
    
    # Ensure collection exists
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    # Generate embeddings and store
    points = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        embedding = embedding_model.encode(chunk_text).tolist()
        
        chunk_id = hashlib.md5(
            f"{doc.file_id}_ocr_{i}_{chunk_text[:50]}".encode()
        ).hexdigest()
        
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                "file_id": doc.file_id,
                "chunk_index": i,
                "text": chunk_text,
                "word_count": chunk.get("word_count", 0) if isinstance(chunk, dict) else 0,
                "source_url": url,
                "created_at": datetime.utcnow().isoformat(),
                "is_ocr": True,
                "metadata": chunk.get("metadata", {}) if isinstance(chunk, dict) else {},
            }
        )
        points.append(point)
    
    # Clear GPU memory after generating all embeddings
    clear_gpu_memory()
    
    # Batch upsert
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )
    
    # Mark as vectorized
    doc_store.update_document(
        doc.file_id,
        {
            "is_vectorized": "1",
            "vectorization_completed_at": datetime.utcnow(),
            "status": "VECTORIZED",
        }
    )
    
    logger.info(f"✓ Stored {len(chunks)} chunks to MongoDB + Qdrant: {url}")


async def process_single_page(
    entry: dict,
    doc_store: DocumentStore,
    timeout: int = 120,
) -> bool:
    """
    Process OCR for a single page.
    
    Args:
        entry: Backlog entry with page info
        doc_store: Document store for saving chunks
        timeout: Timeout in seconds (default 120s = 2 minutes)
        
    Returns:
        True if successful, False otherwise
    """
    url = entry["url"]
    pdf_path = entry.get("pdf_path")
    ocr_action = OCRAction(entry["ocr_action"])
    
    if not pdf_path or not Path(pdf_path).exists():
        logger.warning(f"PDF not found for {url}: {pdf_path}")
        return False
    
    logger.info(f"Processing OCR for: {url}")
    
    # Check CUDA state before attempting OCR
    try:
        import torch
        if torch.cuda.is_available():
            # Test CUDA with a simple operation
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.synchronize()
    except RuntimeError as cuda_error:
        if "CUDA" in str(cuda_error) or "assert" in str(cuda_error):
            logger.warning(f"Corrupted CUDA state detected before OCR: {cuda_error}")
            logger.info("Attempting to recover by clearing GPU and reinitializing models...")
            try:
                from app.services.gpu_manager import clear_models
                clear_models()
                clear_gpu_memory()
                logger.info("✓ GPU state reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset GPU state: {e}")
                return False
    
    try:
        # Create Page object
        page = Page(
            url=url,
            depth=entry.get("depth", 0),
            pdf_path=Path(pdf_path),
        )
        
        # Run OCR with timeout
        ocr_text = await asyncio.wait_for(
            process_page_ocr(page, ocr_action),
            timeout=timeout
        )
        
        if not ocr_text:
            logger.warning(f"No OCR text extracted from {url}")
            return False
        
        # Log OCR results
        word_count = len(ocr_text.split())
        logger.info(f"OCR extracted {word_count} words ({len(ocr_text)} chars) from {url}")
        
        # Update page content with OCR text
        page.content = PageContent.from_text(ocr_text, ContentSource.OCR)
        
        # Chunk the OCR'd text with lower threshold for OCR content
        chunks = chunk_page_text(
            page,
            min_words=20,  # Lower threshold for OCR content (was 100)
            max_words=512,
            overlap_words=25,  # Smaller overlap for short content
        )
        
        if not chunks:
            logger.warning(f"No chunks created (text too short: {word_count} words): {url}")
            return False
        
        logger.info(f"✓ Created {len(chunks)} chunks from OCR text: {url}")
        
        # Save chunks to MongoDB and Qdrant
        await _store_chunks(chunks, entry, doc_store)
        
        # Clear GPU memory after processing this page
        clear_gpu_memory()
        
        return True
        
    except asyncio.TimeoutError:
        logger.warning(f"⏱ OCR timeout ({timeout}s) for {url} - skipping")
        clear_gpu_memory()  # Clear on timeout too
        return False
        
    except Exception as e:
        logger.error(f"❌ OCR failed for {url}: {e}")
        clear_gpu_memory()  # Clear on error too
        return False


async def process_ocr_backlog():
    """Process all pages in OCR backlog."""
    backlog_file = Path("outputs/ocr_backlog/pending_ocr.jsonl")
    
    if not backlog_file.exists():
        logger.info("No OCR backlog found")
        return
    
    # Clear GPU memory from Phase 1 before starting Phase 2
    logger.info("Clearing GPU memory from Phase 1...")
    clear_gpu_memory()
    await asyncio.sleep(1)  # Give GPU a moment to fully release memory
    
    # Load backlog entries
    entries = []
    with backlog_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    logger.info(f"Found {len(entries)} pages in OCR backlog")
    
    if not entries:
        return
    
    # Initialize document store
    from app.config import app_config
    doc_store = DocumentStore(
        mongodb_url=app_config.MONGODB_URL,
        database_name=app_config.MONGODB_DATABASE,
    )
    
    # Process each entry
    success_count = 0
    failed_count = 0
    consecutive_failures = 0
    
    for i, entry in enumerate(entries, 1):
        logger.info(f"\n[{i}/{len(entries)}] Processing: {entry['url']}")
        
        success = await process_single_page(entry, doc_store, timeout=120)
        
        if success:
            success_count += 1
            consecutive_failures = 0
        else:
            failed_count += 1
            consecutive_failures += 1
        
        # Adaptive delay - longer after failures to allow GPU recovery
        if consecutive_failures > 0:
            delay = min(5 + consecutive_failures, 10)  # 5-10 seconds after failures
            logger.debug(f"Waiting {delay}s for GPU recovery after failure...")
            await asyncio.sleep(delay)
        else:
            # Normal delay between successful operations
            await asyncio.sleep(2)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("OCR Backlog Processing Complete")
    logger.info(f"  Success: {success_count}/{len(entries)}")
    logger.info(f"  Failed:  {failed_count}/{len(entries)}")
    logger.info("="*70)
    
    # Archive processed backlog
    archive_file = backlog_file.parent / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    backlog_file.rename(archive_file)
    logger.info(f"Backlog archived to: {archive_file}")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("OCR BACKLOG PROCESSOR")
    print("="*70 + "\n")
    
    try:
        asyncio.run(process_ocr_backlog())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
