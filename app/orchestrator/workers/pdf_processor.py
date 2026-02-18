"""
PDF Processor Worker

Processes PDF files using Docling (GPU) to extract text.
GPU-intensive worker - should have minimal concurrency (1-2 workers max).
"""

import aiofiles
import asyncio
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.config import get_logger, OCR_MIN_WORD_COUNT_SUFFICIENT
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import PdfTask, StorageTask
from app.docling.processor import AsyncDocumentProcessor
from app.crawling.models.document import Page, PageContent, ContentSource
from docling_core.transforms.chunker.base import BaseChunk

logger = get_logger(__name__)


class PdfProcessorWorker(BaseWorker):
    """
    PDF processor worker - handles GPU-based PDF text extraction using Docling.
    
    Responsibilities (GPU):
    - Pull PdfTask from pdf_queue
    - Extract text from PDF using Docling (GPU)
    - Chunk the extracted text
    - Push chunks to storage_queue
    - If extraction fails or minimal text: mark for OCR backlog
    
    IMPORTANT: Only 1-2 workers should run to prevent GPU overload!
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        timeout_seconds: int = 120,
    ):
        super().__init__(worker_id, queue_manager)
        self.timeout_seconds = timeout_seconds
        
        # Docling processor (lazy initialization)
        self._docling_processor = None
        
        # Statistics
        self.total_processing_time = 0.0
        self.pdfs_processed = 0
        self.pdfs_failed = 0
    
    @property
    def worker_type(self) -> str:
        return "pdf_processor"
    
    async def startup(self):
        """Initialize PDF processor resources (loads GPU models)."""
        await super().startup()
        logger.info(f"[{self.worker_id}] PDF processor worker initialized (GPU)")
    
    async def get_next_task(self) -> Optional[PdfTask]:
        """Get next PDF task from queue."""
        return await self.queue_manager.get_pdf_task(timeout=2.0)
    
    async def process_task(self, task: PdfTask) -> bool:
        """
        Extract text from PDF using Docling and prepare for storage.
        
        Args:
            task: PdfTask with PDF to process
            
        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)
        
        page = task.page
        pdf_path = page.pdf_path
        
        if not pdf_path or not pdf_path.exists():
            logger.error(f"[{self.worker_id}] PDF not found: {pdf_path}")
            task.mark_failed("PDF file not found")
            return False
        
        logger.info(f"[{self.worker_id}] Processing PDF: {page.url}")
        
        start_time = time.time()
        
        try:
            # Extract text from PDF with timeout
            text = await asyncio.wait_for(
                self._extract_pdf_text(pdf_path),
                timeout=self.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.pdfs_processed += 1
            
            word_count = len(text.split()) if text else 0
            if text and word_count > OCR_MIN_WORD_COUNT_SUFFICIENT:  # Has meaningful text
                # Update page content
                page.content = PageContent.from_text(text, ContentSource.TEXT_LAYER)
                
                logger.info(
                    f"[{self.worker_id}] Extracted {len(text)} chars ({word_count} words) from PDF "
                    f"in {processing_time:.2f}s: {page.url}"
                )
                
                # Optional: Save as markdown file
                file_id = page.url_hash if hasattr(page, 'url_hash') else pdf_path.stem
                markdown_path = await self._save_markdown(pdf_path, file_id)
                if markdown_path:
                    logger.info(f"[{self.worker_id}] Saved markdown: {markdown_path}")
                
                # Create chunks using Docling's HybridChunker
                chunks = await self._create_chunks_from_docling()
                
                if chunks:
                    # Create storage task
                    storage_task = StorageTask(
                        task_id=f"{task.task_id}_storage",
                        website_url=task.website_url,
                        crawl_session_id=task.crawl_session_id,
                        chunks=chunks,
                        document_metadata={
                            "url": page.url,
                            "depth": page.depth,
                            "pdf_processed": True,
                            "docling_chunked": True,
                            "processing_time_seconds": processing_time,
                        },
                        priority=task.priority,
                    )
                    
                    await self.queue_manager.put_storage_task(storage_task)
                    
                    logger.info(
                        f"[{self.worker_id}] Created {len(chunks)} HybridChunker chunks "
                        f"from PDF: {page.url}"
                    )
                else:
                    logger.warning(f"[{self.worker_id}] No chunks created from PDF: {page.url}")
                
                task.mark_completed()
                return True
                
            else:
                # Minimal or no text - likely scanned PDF
                word_count = len(text.split()) if text else 0
                logger.warning(
                    f"[{self.worker_id}] Minimal text extracted from PDF ({word_count} words, threshold: {OCR_MIN_WORD_COUNT_SUFFICIENT}). "
                    f"Marking for OCR: {page.url}"
                )
                
                # Save to OCR backlog
                await self._save_to_ocr_backlog(page, task.website_url)
                
                task.mark_completed()
                return True
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(
                f"[{self.worker_id}] PDF processing timeout ({self.timeout_seconds}s) "
                f"for {page.url} - marking for OCR"
            )
            
            # Timeout - save to OCR backlog
            await self._save_to_ocr_backlog(page, task.website_url)
            
            task.mark_completed()
            return True  # Don't fail the task, just skip Docling
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.pdfs_failed += 1
            
            logger.error(
                f"[{self.worker_id}] PDF processing failed for {page.url} "
                f"after {processing_time:.2f}s: {e}",
                exc_info=True
            )
            
            # Try to save to OCR backlog as fallback
            try:
                await self._save_to_ocr_backlog(page, task.website_url)
                logger.info(f"[{self.worker_id}] Saved to OCR backlog as fallback: {page.url}")
                task.mark_completed()
                return True
            except:
                pass
            
            task.mark_failed(str(e))
            return False
    
    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using Docling and initialize for chunking.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text (markdown preview)
        """
        # Lazy initialization of Docling processor with GPU
        if self._docling_processor is None:
            self._docling_processor = AsyncDocumentProcessor(
                use_gpu=True,  # Enable GPU for Phase 1 PDF processing
                executor=None,  # Will use executor from manager
                skip_ocr=True,  # CRITICAL: Disable OCR in Phase 1 to prevent memory exhaustion
            )
            # Initialize chunker components
            await self._docling_processor.initialize_async()
        
        # Convert document (this sets self._docling_processor.doc)
        result = await self._docling_processor.convert_document_async(
            str(pdf_path)
        )
        
        # Extract text from the DoclingDocument for preview/validation
        if result and hasattr(result, 'export_to_markdown'):
            return result.export_to_markdown()
        
        return ""
    
    async def _create_chunks_from_docling(self) -> List[Dict[str, Any]]:
        """
        Create chunks using Docling's HybridChunker and convert to storage format.
        
        Returns:
            List of chunk dictionaries ready for storage
        """
        if not self._docling_processor or not self._docling_processor.doc:
            logger.error(f"[{self.worker_id}] Cannot create chunks - document not converted")
            return []
        
        try:
            # Create chunks using HybridChunker
            base_chunks = await self._docling_processor.create_chunks_async()
            
            if not base_chunks:
                logger.warning(f"[{self.worker_id}] HybridChunker returned no chunks")
                return []
            
            # Convert BaseChunk objects to storage-compatible dictionaries
            storage_chunks = []
            for idx, chunk in enumerate(base_chunks):
                # Contextualize chunk to get full text
                chunk_text = await self._contextualize_chunk(chunk)
                
                # Extract metadata from chunk
                chunk_dict = {
                    "text": chunk_text,
                    "chunk_index": idx,
                    "word_count": len(chunk_text.split()),
                    "metadata": {
                        "page_number": self._extract_page_number(chunk),
                        "heading_text": self._extract_heading(chunk),
                        "doc_items_refs": self._extract_doc_items_refs(chunk),
                        "has_image": "![" in chunk_text and "](" in chunk_text,
                        "source": "docling_hybrid_chunker",
                    }
                }
                storage_chunks.append(chunk_dict)
            
            logger.debug(
                f"[{self.worker_id}] Converted {len(storage_chunks)} BaseChunks to storage format"
            )
            return storage_chunks
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to create chunks from Docling: {e}",
                exc_info=True
            )
            return []
    
    async def _contextualize_chunk(self, chunk: BaseChunk) -> str:
        """Contextualize a chunk to get its full text representation."""
        try:
            if not self._docling_processor or not self._docling_processor.chunker:
                # Fallback to chunk.text if chunker not available
                return chunk.text if hasattr(chunk, 'text') else str(chunk)
            
            # Run contextualization in executor
            loop = asyncio.get_event_loop()
            ctx_text = await loop.run_in_executor(
                self._docling_processor.executor,
                lambda: self._docling_processor.chunker.contextualize(chunk=chunk)
            )
            return ctx_text
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Contextualization failed: {e}")
            # Fallback to chunk.text
            return chunk.text if hasattr(chunk, 'text') else str(chunk)
    
    def _extract_page_number(self, chunk: BaseChunk) -> int:
        """Extract page number from chunk metadata."""
        try:
            from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
            doc_chunk = DocChunk.model_validate(chunk)
            if doc_chunk.meta and doc_chunk.meta.doc_items:
                for item in doc_chunk.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                return prov.page_no
        except Exception:
            pass
        return 1
    
    def _extract_heading(self, chunk: BaseChunk) -> Optional[str]:
        """Extract heading from chunk metadata."""
        try:
            from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
            doc_chunk = DocChunk.model_validate(chunk)
            if doc_chunk.meta and doc_chunk.meta.headings:
                return " > ".join(doc_chunk.meta.headings)
        except Exception:
            pass
        return None
    
    def _extract_doc_items_refs(self, chunk: BaseChunk) -> List[str]:
        """Extract document item references from chunk."""
        try:
            from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
            doc_chunk = DocChunk.model_validate(chunk)
            if doc_chunk.meta and doc_chunk.meta.doc_items:
                return [str(item.self_ref) for item in doc_chunk.meta.doc_items]
        except Exception:
            pass
        return []
    
    async def _save_markdown(self, pdf_path: Path, file_id: str) -> Optional[Path]:
        """
        Save already-converted PDF as markdown file.
        
        Args:
            pdf_path: Path to PDF file (for reference)
            file_id: Unique identifier for the file
            
        Returns:
            Path to saved markdown file, or None if failed
        """
        try:
            # Use already converted document
            if not self._docling_processor or not self._docling_processor.doc:
                logger.warning(f"[{self.worker_id}] Cannot save markdown - document not converted")
                return None
            
            # Export to markdown
            result = self._docling_processor.doc
            if result and hasattr(result, 'export_to_markdown'):
                markdown_content = result.export_to_markdown()
                
                # Create markdown output directory
                markdown_dir = Path("outputs/markdown")
                markdown_dir.mkdir(parents=True, exist_ok=True)
                
                # Save markdown file
                markdown_path = markdown_dir / f"{file_id}.md"
                async with aiofiles.open(markdown_path, 'w', encoding='utf-8') as f:
                    await f.write(markdown_content)
                
                return markdown_path
            
            return None
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to save markdown: {e}", exc_info=True)
            return None
    
    async def _save_to_ocr_backlog(self, page: Page, website_url: str):
        """Save page to OCR backlog for Phase 2 processing."""
        import json
        from datetime import datetime
        
        backlog_dir = Path("outputs/ocr_backlog")
        backlog_dir.mkdir(parents=True, exist_ok=True)
        
        backlog_file = backlog_dir / "pending_ocr.jsonl"
        
        entry = {
            "url": page.url,
            "pdf_path": str(page.pdf_path) if page.pdf_path else None,
            "ocr_action": "full_page_ocr",
            "website_url": website_url,
            "depth": page.depth,
            "timestamp": datetime.utcnow().isoformat(),
            "url_hash": page.url_hash if hasattr(page, 'url_hash') else None,
            "reason": "docling_extraction_failed",
        }
        
        # Append to JSONL file
        with backlog_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        logger.info(f"[{self.worker_id}] Saved to OCR backlog: {page.url}")
    
    @property
    def stats(self) -> dict:
        """Get PDF processor worker statistics."""
        base_stats = super().stats
        
        avg_processing_time = (
            self.total_processing_time / self.pdfs_processed
            if self.pdfs_processed > 0 else 0
        )
        
        base_stats.update({
            "pdfs_processed": self.pdfs_processed,
            "pdfs_failed": self.pdfs_failed,
            "total_processing_time_seconds": self.total_processing_time,
            "avg_processing_time_seconds": avg_processing_time,
        })
        
        return base_stats
