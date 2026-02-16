######check markdown appearance of images with async/await######

import json
import logging
import aiofiles
from app.docling.serializers.placeholder import ImgPlaceholderSerializerProvider
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from docling.chunking import HybridChunker

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer, MarkdownListSerializer, MarkdownPictureSerializer,
    MarkdownInlineSerializer, MarkdownTextSerializer, MarkdownParams,
    MarkdownKeyValueSerializer, MarkdownFormSerializer, MarkdownMetaSerializer,
    MarkdownAnnotationSerializer
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from typing import Iterable, Optional, List, Tuple
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.labels import DocItemLabel
from rich.console import Console
from rich.panel import Panel

from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRefMode,
    PictureDescriptionData,
    PictureItem,
)
from docling_surya import SuryaOcrOptions
from pathlib import Path
from app.config import app_config
from app.docling.qdrant_service import DoclingQdrantService
from app.docling.qdrant_adapter import DoclingQdrantAdapter
from app.core.executor import executor_manager

logger = logging.getLogger("docling_document_processor")
class AsyncDocumentProcessor:
    """Async document processor for chunking and analyzing documents with images"""
    
    def __init__(
        self,
       
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: Path = Path("exported_images"),
        console_width: int = 200,
        use_gpu: bool = True,
        executor: Optional[ThreadPoolExecutor] = None,
        serializer_provider: Optional[ImgPlaceholderSerializerProvider] = None,
        qdrant_service: DoclingQdrantService = None,
        docling_qdrant_adapter: DoclingQdrantAdapter = None,
        skip_ocr: bool = False,  # Disable OCR for Phase 1 orchestrator
    ):
        
        self.embed_model_id = embed_model_id
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.skip_ocr = skip_ocr  # Store OCR preference
        # Use provided executor or None (will be acquired from manager)
        self.executor = executor
        self._executor_acquired = False  # Track if we acquired from manager
        
        # Initialize console
        self.console = Console(width=console_width)
        
        # Document and processing objects
        self.doc: Optional[DoclingDocument] = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.chunker: Optional[HybridChunker] = None
        self.chunks: List[BaseChunk] = []
        self.serializer_provider=serializer_provider
        self.qdrant_service=qdrant_service
        self.docling_qdrant_adapter=docling_qdrant_adapter
        
        # Results
        self.image_chunks: List[int] = []
        
        # CRITICAL FIX: Cache DocumentConverter to prevent re-initialization
        self._document_converter: Optional[DocumentConverter] = None
        self._converter_lock = threading.Lock()  # Thread-safe lock for executor
    
    @classmethod
    async def from_config(cls,_executor: Optional[ThreadPoolExecutor] = None):
        """Initialize Enhanced DocumentProcessor with all required dependencies"""
        # Pass executor to serializer provider so it can be shared
        img_serializer_service = await ImgPlaceholderSerializerProvider.from_config(_executor=_executor)
        qdrant_service = await DoclingQdrantService.from_config()
        docling_qdrant_adapter = await DoclingQdrantAdapter.from_config()
        output_dir=app_config.exported_images
        use_gpu = getattr(app_config, 'USE_GPU', True)  # Get GPU setting from config
        
        return cls(
            output_dir=Path(output_dir),
            executor=_executor,
            serializer_provider=img_serializer_service,
            qdrant_service=qdrant_service,
            docling_qdrant_adapter=docling_qdrant_adapter,
            use_gpu=use_gpu
        )


    
    async def initialize_async(self):
        """Initialize async components"""
        # Acquire executor from manager if not provided
        if self.executor is None:
            self.executor = await executor_manager.acquire()
            self._executor_acquired = True
            logger.debug("Acquired executor from manager")
        
        loop = asyncio.get_event_loop()
        
        # Initialize tokenizer in executor
        self.tokenizer = await loop.run_in_executor(
            self.executor,
            lambda: HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.embed_model_id)
            )
        )
        
        # # Initialize serializer provider
        # self.serializer_provider = ImgPlaceholderSerializerProvider(
        #     output_dir=self.output_dir,
        #     executor=self.executor
        # )
        
        return self
    
    def _configure_pipeline_options(self) -> PdfPipelineOptions:
        """Configure PDF pipeline options"""
        pipeline_options = PdfPipelineOptions()
        
        # OCR settings - disabled in orchestrator Phase 1, enabled in Phase 2
        pipeline_options.do_ocr = not self.skip_ocr
        
        if pipeline_options.do_ocr:
            pipeline_options.ocr_options = SuryaOcrOptions(
                force_full_page_ocr=False,  # Only OCR images, not full pages (faster)
                use_gpu=self.use_gpu
            )
        
        # Feature settings
        pipeline_options.do_code_enrichment = False  # Disable if not needed (faster)
        pipeline_options.do_formula_enrichment = False  # Disable if not needed (faster)
        pipeline_options.do_table_structure = True  # Set to False if tables not needed (20% faster)
        pipeline_options.images_scale = 1.0  # Lower = faster (1.0 is fastest, 1.5 balanced, 2.0 quality)
        pipeline_options.generate_page_images = False  # Disable if not needed (faster)
        pipeline_options.allow_external_plugins = True  # Enable to use surya-ocr plugin
        
        logger.info(f"Pipeline options: OCR={pipeline_options.do_ocr}, GPU={self.use_gpu}")
        return pipeline_options
    
    def _get_or_create_converter_sync(self) -> DocumentConverter:
        """Get cached DocumentConverter or create new one (thread-safe synchronous)."""
        with self._converter_lock:
            if self._document_converter is None:
                logger.info("Initializing DocumentConverter (ONE TIME)...")
                pipeline_options = self._configure_pipeline_options()
                
                self._document_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=StandardPdfPipeline,
                            pipeline_options=pipeline_options,
                        ),
                    }
                )
                logger.info("DocumentConverter initialized and cached for reuse")
            
            return self._document_converter
    
    async def convert_document_async(self,pdf_path:str) -> DoclingDocument:
        """Convert PDF document asynchronously"""
        # Check file size
        from pathlib import Path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        logger.info(f"Starting document conversion: {pdf_path} ({file_size_mb:.2f} MB)")
        
        # Warn if file is large
        if file_size_mb > 50:
            logger.warning(f"Large PDF detected ({file_size_mb:.2f} MB). This may take several minutes...")
        
        # Run conversion in executor with proper thread-safe serialization
        loop = asyncio.get_event_loop()
        
        from app.core.timeouts import TimeoutConfig
        timeout_config = TimeoutConfig()
        
        # Wrapper function that acquires converter inside the thread
        def _convert():
            # Get converter with thread-safe lock
            converter = self._get_or_create_converter_sync()
            return converter.convert(source=pdf_path)
        
        try:
            # Increase timeout for large PDFs (scale with file size)
            timeout = min(timeout_config.PDF_PROCESSING * (1 + file_size_mb / 50), 600)
            
            # Run conversion in executor with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, _convert),
                timeout=timeout
            )
                
            self.doc = result.document
            logger.info(f"Document converted successfully: {pdf_path} ({file_size_mb:.2f} MB)")
            return self.doc
            
        except asyncio.TimeoutError:
            logger.error(f"Document conversion timed out after {timeout:.1f}s: {pdf_path} ({file_size_mb:.2f} MB)")
            raise TimeoutError(f"Document conversion took too long (>{timeout:.1f}s): {pdf_path}")
        except Exception as e:
            logger.error(f"Document conversion failed: {e}", exc_info=True)
            raise
        finally:
            # Always cleanup GPU memory after conversion attempt
            await self._cleanup_gpu_memory()
    
    async def create_chunks_async(self) -> List[BaseChunk]:
        """Create chunks asynchronously"""
        if not self.doc:
            raise ValueError("Document not converted. Call convert_document_async() first.")
        
        if not self.tokenizer or not self.serializer_provider:
            raise ValueError("Not initialized. Call initialize_async() first.")
        
        # Create chunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            serializer_provider=self.serializer_provider,
        )
        
        # Run chunking in executor
        loop = asyncio.get_event_loop()
        chunk_iter = await loop.run_in_executor(
            self.executor,
            lambda: self.chunker.chunk(dl_doc=self.doc)
        )
        
        self.chunks = list(chunk_iter)
        
        # Filter out non-English chunks
        self.chunks = await self._filter_english_chunks(self.chunks)
        
        return self.chunks
    
    async def _filter_english_chunks(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """Filter chunks to keep only English content"""
        try:
            from langdetect import detect, LangDetectException
        except ImportError:
            logger.warning("langdetect not installed, skipping language filtering")
            return chunks
        
        english_chunks = []
        filtered_count = 0
        
        for chunk in chunks:
            # Get chunk text
            chunk_text = chunk.text.strip() if hasattr(chunk, 'text') else str(chunk).strip()
            
            # Skip empty chunks
            if not chunk_text or len(chunk_text) < 10:
                english_chunks.append(chunk)
                continue
            
            try:
                # Detect language
                lang = detect(chunk_text)
                
                if lang == 'en':
                    english_chunks.append(chunk)
                else:
                    logger.debug(f"Filtered non-English chunk ({lang}): {chunk_text[:50]}...")
                    filtered_count += 1
            except LangDetectException:
                # If detection fails, keep the chunk (might be technical content)
                english_chunks.append(chunk)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} non-English chunks, kept {len(english_chunks)} English chunks")
        
        return english_chunks

    async def analyze_chunk_async(self, chunk_pos: int) -> Tuple[bool, str, int]:
        """Analyze a single chunk asynchronously"""
        if not self.chunks or chunk_pos >= len(self.chunks):
            raise ValueError(f"Invalid chunk position: {chunk_pos}")
        
        chunk = self.chunks[chunk_pos]
        
        # Run contextualization in executor
        loop = asyncio.get_event_loop()
        ctx_text = await loop.run_in_executor(
            self.executor,
            lambda: self.chunker.contextualize(chunk=chunk)
        )
        
        # Count tokens
        num_tokens = await loop.run_in_executor(
            self.executor,
            lambda: self.tokenizer.count_tokens(text=ctx_text)
        )
        
        # Check for images
        has_image = "![" in ctx_text and "](" in ctx_text
        
        return has_image, ctx_text, num_tokens
    
    async def print_chunk_async(self, chunk_pos: int):
        """Print a single chunk asynchronously"""
        chunk = self.chunks[chunk_pos]
        has_image, ctx_text, num_tokens = await self.analyze_chunk_async(chunk_pos)
        
        doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
        image_indicator = " IMAGE" if has_image else ""
        
        title = f"chunk_pos={chunk_pos} num_tokens={num_tokens} doc_items_refs={doc_items_refs}{image_indicator}"
        self.console.print(Panel(ctx_text, title=title))
    
    async def process_all_chunks_async(self, print_chunks: bool = True) -> List[int]:
        """Process all chunks and identify image chunks"""
        self.image_chunks = []
        
        # Process chunks concurrently in batches
        batch_size = 10
        for i in range(0, len(self.chunks), batch_size):
            batch_end = min(i + batch_size, len(self.chunks))
            batch_tasks = []
            
            for chunk_pos in range(i, batch_end):
                batch_tasks.append(self.analyze_chunk_async(chunk_pos))
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Process results and print
            for chunk_pos, (has_image, ctx_text, num_tokens) in enumerate(batch_results, start=i):
                if has_image:
                    self.image_chunks.append(chunk_pos)
                    self.console.print(f"\n✓ CHUNK {chunk_pos} CONTAINS IMAGE(S):")
                    self.console.print("-" * 100)
                
                if print_chunks:
                    await self.print_chunk_async(chunk_pos)
        
        return self.image_chunks
    
    def find_n_th_chunk_with_label(
        self, n: int, label: DocItemLabel
    ) -> Tuple[Optional[int], Optional[BaseChunk]]:
        """Find the n-th chunk with a specific label"""
        num_found = -1
        for i, chunk in enumerate(self.chunks):
            doc_chunk = DocChunk.model_validate(chunk)
            for it in doc_chunk.meta.doc_items:
                if it.label == label:
                    num_found += 1
                    if num_found == n:
                        return i, chunk
        return None, None
    

    
    async def _save_chunks_locally(self, file_id: str, file_name: str, chunks_metadata: list):
        """Save chunks to local JSON file for verification"""
        try:
            from pathlib import Path
            
            # Create chunks output directory
            chunks_dir = Path(app_config.OUTPUT_DIR) / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            output_file = chunks_dir / f"{file_id}.json"
            
            # Convert to serializable format
            chunks_data = {
                "file_id": file_id,
                "file_name": file_name,
                "total_chunks": len(chunks_metadata),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,  # Preview
                        "full_text_length": len(chunk.text),
                        "token_count": chunk.token_count,
                        "has_image": chunk.has_image,
                        "heading_text": chunk.heading_text,
                        "doc_items_refs": chunk.doc_items_refs
                    }
                    for chunk in chunks_metadata
                ]
            }
            
            # Use aiofiles for async file write
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(chunks_data, indent=2, ensure_ascii=False))
            
            print(f"[INFO] Saved {len(chunks_metadata)} chunks to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks locally: {e}", exc_info=True)
    
    async def _cleanup_gpu_memory(self):
        """Force GPU memory cleanup - safe to call multiple times"""
        if not self.use_gpu:
            return
            
        try:
            import torch
            import gc
            
            # First clear Python references
            gc.collect()
            
            # Then clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("GPU memory cleared successfully")
                except RuntimeError as cuda_err:
                    # Ignore CUDA errors during cleanup (e.g., OOM)
                    logger.warning(f"CUDA cleanup warning: {cuda_err}")
                except Exception as e:
                    logger.error(f"GPU cleanup error: {e}")
        except ImportError:
            # torch not available
            pass
        except Exception as e:
            logger.error(f"Failed to cleanup GPU memory: {e}", exc_info=True)
    
    async def close(self):
        """Cleanup resources and free GPU memory (idempotent - safe to call multiple times)"""
        # Check if already closed
        if self.executor is None and not self._executor_acquired:
            logger.debug("Processor already closed, skipping cleanup")
            return
        
        try:
            # Close serializer provider
            if self.serializer_provider:
                try:
                    await self.serializer_provider.close()
                except Exception as e:
                    logger.error(f"Error closing serializer provider: {e}")
            
            # Clear document references
            self.doc = None
            self.chunks = []
            self.chunker = None
            
            # Force GPU memory cleanup
            await self._cleanup_gpu_memory()
            
            # Release executor if we acquired it from manager
            if self._executor_acquired and self.executor is not None:
                await executor_manager.release()
                self.executor = None
                self._executor_acquired = False
                logger.debug("Released executor to manager")
            
            logger.debug("Processor resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during processor cleanup: {e}", exc_info=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def get_contextualized_chunks_async(self) -> List[tuple]:
        """Get all chunks with their contextualized text and token counts"""
        if not self.chunks:
            raise ValueError("No chunks available. Call create_chunks_async() first.")
        
        results = []
        batch_size = 10
        print_chunks=True
        for i in range(0, len(self.chunks), batch_size):
            batch_end = min(i + batch_size, len(self.chunks))
            batch_tasks = []
            
            for chunk_pos in range(i, batch_end):
                batch_tasks.append(self.analyze_chunk_async(chunk_pos))
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            for chunk_pos, (has_image, ctx_text, num_tokens) in enumerate(batch_results, start=i):
                results.append((self.chunks[chunk_pos], ctx_text, num_tokens))

                if print_chunks:
                    await self.print_chunk_async(chunk_pos)

                    chunk = self.chunks[chunk_pos]
                    # has_image, ctx_text, num_tokens = await self.analyze_chunk_async(chunk_pos)
                            
                    doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
                    image_indicator = " IMAGE" if has_image else ""
                    
                    title = f"chunk_pos={chunk_pos} num_tokens={num_tokens} doc_items_refs={doc_items_refs}{image_indicator}"
                    self.console.print(Panel(ctx_text, title=title))
        
        return results
    



    # Main execution function
    async def process_document_main(self,file_id:str,file_name:str,pdfpath:str):

        """Main async function to process document"""
        try:

            await self.initialize_async()

            print(" Converting document...")
            await self.convert_document_async(pdfpath)
            print(" Document converted")
                
                # Create chunks
            print("\n Creating chunks...")
            await self.create_chunks_async()
            print(f" Created {len(self.chunks)} chunks")
                
            print(f"[INFO] Contextualizing chunks...")
            chunk_data = await self.get_contextualized_chunks_async()

                        # Prepare for Qdrant
            chunks, texts, tokens = zip(*chunk_data)
            chunks_metadata = self.docling_qdrant_adapter.prepare_chunks_for_qdrant(
                    list(chunks),
                    list(texts),
                    list(tokens)
                )
            
            # Save chunks locally for verification
            print(f"[INFO] Saving {len(chunks_metadata)} chunks locally...")
            await self._save_chunks_locally(file_id, file_name, chunks_metadata)
            
                # Insert into Qdrant
            print(f"[INFO] Inserting {len(chunks_metadata)} chunks into Qdrant...")
            results = await self.qdrant_service.insert_docling_chunks(
                    chunks_metadata=chunks_metadata,
                    file_id=file_id,
                    filename=file_name,
                    check_duplicates=True  # Enable duplicate detection
                )
                
            print(f"[INFO] Qdrant insertion results: {results}")
                
            if results > 0:
                print(f"[SUCCESS] Vector inserted successfully for {file_id}")
                return file_id
            else:
                print(f"[No] Vector inserted  for {file_id}")
                logger.error(f"No vectors inserted for {file_id}")
                return None
                
        except Exception as e:
            print(f"Pipeline Vector inserted  for {file_name}: {e}")
            logger.error(f"Pipeline failed for {file_name}: {e}", exc_info=True)
            return None        

                   
        
  
