import logging
import os
from app.bot.processdocling.imgplaceholder_serializer import ImgPlaceholderSerializerProvider
from app.bot.processdocling.pictureserializer import PicturePostProcessor
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorDevice, AcceleratorOptions
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

from typing import Dict, Iterable, Optional, List, Tuple
import asyncio
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
from app.bot.processdocling.docling_qdrant_service import DoclingQdrantService
from app.bot.processdocling.docling_qdrant_adapter import DoclingQdrantAdapter
from app.bot.translate.languagedetector import LanguageDetector
import re
logger = logging.getLogger("docling_document_processor")


def _detect_gpu_availability() -> bool:
    """
    Detect if GPU is available and working for document processing.
    Supports: NVIDIA Tesla T4 (Azure NC16as T4 v3), sm_75 Turing architecture
    Falls back to CPU if GPU is unavailable or incompatible.
    
    Environment variable USE_GPU can override auto-detection:
    - USE_GPU=true: Force GPU (will fail if not available)
    - USE_GPU=false: Force CPU
    - USE_GPU not set: Auto-detect
    """
    use_gpu_env = os.environ.get("USE_GPU", "").lower()
    if use_gpu_env == "false":
        logger.info("GPU disabled via USE_GPU=false environment variable, using CPU")
        return False
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return False
        
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_capability = torch.cuda.get_device_capability(0)
        cuda_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]}"
        
        logger.info(f"GPU detected: {device_name} (compute capability: {cuda_arch}, devices: {device_count})")
        
        try:
            test_tensor = torch.zeros(1, device='cuda:0')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            logger.info(f"GPU is functional, using CUDA on {device_name}")
            print(f"[INFO] GPU enabled: {device_name} ({cuda_arch})")
            return True
        except RuntimeError as cuda_error:
            error_msg = str(cuda_error)
            if "no kernel image is available" in error_msg:
                logger.warning(f"GPU {device_name} ({cuda_arch}) not supported by current PyTorch")
            else:
                logger.warning(f"CUDA operations failed: {cuda_error}")
            logger.info("Falling back to CPU processing")
            return False
            
    except ImportError:
        logger.warning("PyTorch not installed, using CPU")
        return False
    except Exception as e:
        logger.warning(f"GPU detection error: {e}, falling back to CPU")
        return False


class AsyncDocumentProcessor:
    """Async document processor for chunking and analyzing documents with images"""
    
    def __init__(
        self,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: Path = Path("exported_images"),
        console_width: int = 200,
        use_gpu: bool = None,
        executor: Optional[ThreadPoolExecutor] = None,
        serializer_provider: Optional[ImgPlaceholderSerializerProvider] = None,
        qdrant_service: DoclingQdrantService = None,
        docling_qdrant_adapter: DoclingQdrantAdapter = None,
        picture_processor: PicturePostProcessor = None,

        language_detector: LanguageDetector = None, 
        allowed_languages: List[str] = None,  
        min_language_confidence: float = 0.5, 
        min_text_length_for_detection: int = 20, 
    ):
        self.embed_model_id = embed_model_id
        self.output_dir = output_dir
        
        if use_gpu is None:
            self.use_gpu = _detect_gpu_availability()
        else:
            self.use_gpu = use_gpu
        
        logger.info(f"Document processor initialized with use_gpu={self.use_gpu}")
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        self.console = Console(width=console_width)
        
        self.doc: Optional[DoclingDocument] = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.chunker: Optional[HybridChunker] = None
        self.chunks: List[BaseChunk] = []
        self.serializer_provider = serializer_provider
        self.qdrant_service = qdrant_service
        self.docling_qdrant_adapter = docling_qdrant_adapter
        self.picture_processor = picture_processor


        # Language filtering configuration
        self.language_detector = language_detector 
        self.allowed_languages = allowed_languages or ['en']  
        self.min_language_confidence = min_language_confidence  
        self.min_text_length_for_detection = min_text_length_for_detection  

        
        self.image_chunks: List[int] = []
    
    @classmethod
    async def from_config(cls, _executor: Optional[ThreadPoolExecutor] = None):
        img_serializer_service = await ImgPlaceholderSerializerProvider.from_config()
        qdrant_service = await DoclingQdrantService.from_config()
        docling_qdrant_adapter = await DoclingQdrantAdapter.from_config()
        picture_processor = await PicturePostProcessor.from_config()
        language_detector = await LanguageDetector.languageDetector_from_config()  
        output_dir = app_config.exported_images

                # Get language filtering config from app_config or environment
        allowed_languages = getattr(app_config, 'allowed_chunk_languages', ['en'])  
        min_confidence = getattr(app_config, 'min_language_confidence', 0.5)  
        
        return cls(
            output_dir=Path(output_dir),
            executor=_executor,
            serializer_provider=img_serializer_service,
            qdrant_service=qdrant_service,
            docling_qdrant_adapter=docling_qdrant_adapter,
            picture_processor=picture_processor,
            language_detector=language_detector,
            allowed_languages=allowed_languages,
            min_language_confidence=min_confidence,

        )

    async def initialize_async(self):
        loop = asyncio.get_event_loop()
        self.tokenizer = await loop.run_in_executor(
            self.executor,
            lambda: HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.embed_model_id)
            )
        )
        return self

    def _configure_pipeline_options(self) -> PdfPipelineOptions:
        device = AcceleratorDevice.CUDA if self.use_gpu else AcceleratorDevice.CPU
        device_name = "GPU (CUDA)" if self.use_gpu else "CPU"
        logger.info(f"Configuring docling pipeline with device: {device_name}")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = SuryaOcrOptions(
            force_full_page_ocr=True, ##check this
            use_gpu=self.use_gpu
        )
        
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=device
        )
        
        pipeline_options.do_code_enrichment = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.do_table_structure = True
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = False ## check
        pipeline_options.generate_picture_images = True
        pipeline_options.allow_external_plugins = True
        
        return pipeline_options

    async def convert_document_async(self, pdf_path: str) -> DoclingDocument:
        pipeline_options = self._configure_pipeline_options()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: converter.convert(source=pdf_path)
        )
        self.doc = result.document
        return self.doc

    async def create_chunks_async(self) -> List[BaseChunk]:
        if not self.doc:
            raise ValueError("Document not converted. Call convert_document_async() first.")
        if not self.tokenizer or not self.serializer_provider:
            raise ValueError("Not initialized. Call initialize_async() first.")
        
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            serializer_provider=self.serializer_provider,
        )
        loop = asyncio.get_event_loop()
        chunk_iter = await loop.run_in_executor(
            self.executor,
            lambda: self.chunker.chunk(dl_doc=self.doc)
        )
        self.chunks = list(chunk_iter)
        return self.chunks

    async def analyze_chunk_async(self, chunk_pos: int) -> Tuple[bool, str, int]:
        if not self.chunks or chunk_pos >= len(self.chunks):
            raise ValueError(f"Invalid chunk position: {chunk_pos}")
        
        chunk = self.chunks[chunk_pos]
        loop = asyncio.get_event_loop()
        ctx_text = await loop.run_in_executor(
            self.executor,
            lambda: self.chunker.contextualize(chunk=chunk)
        )
        num_tokens = await loop.run_in_executor(
            self.executor,
            lambda: self.tokenizer.count_tokens(text=ctx_text)
        )
        has_image = "<!-- IMAGE_ID:" in ctx_text
        return has_image, ctx_text, num_tokens

    async def print_chunk_async(self, chunk_pos: int):
        chunk = self.chunks[chunk_pos]
        has_image, ctx_text, num_tokens = await self.analyze_chunk_async(chunk_pos)
        doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
        image_indicator = " IMAGE" if has_image else ""
        title = f"chunk_pos={chunk_pos} num_tokens={num_tokens} doc_items_refs={doc_items_refs}{image_indicator}"
        self.console.print(Panel(ctx_text, title=title))


    async def _detect_chunk_language(self, text: str) -> Tuple[str, float, bool]:
        """
        Detect language of chunk text asynchronously.
        
        Returns:
            Tuple[str, float, bool]: (detected_language, confidence, should_include)
        """
        try:
            # Handle empty or very short text
            if not text or len(text.strip()) < self.min_text_length_for_detection:
                logger.debug(f"Text too short for language detection ({len(text)} chars), assuming English")
                return 'en', 1.0, True
            
            # Clean text for detection (remove markdown, placeholders, etc.)
            cleaned_text = self._clean_text_for_language_detection(text)
            
            if not cleaned_text or len(cleaned_text.strip()) < self.min_text_length_for_detection:
                logger.debug("Text empty after cleaning, assuming English")
                return 'en', 1.0, True
            
            # Run language detection in executor (blocking operation)
            # loop = asyncio.get_event_loop()
            # detected_lang, confidence, is_mixed, mixed_info = await loop.run_in_executor(
            #     self.executor,
            #     lambda: self.language_detector.detect_language(cleaned_text)
            # )
            # CORRECT - call async method directly with await
            detected_lang, confidence, is_mixed, mixed_info = await self.language_detector.detect_language(cleaned_text)
            
            # Map to ISO code (if it's in indic format)
            iso_lang = detected_lang
            if detected_lang in self.language_detector.INDIC_LANG_MAPPING:
                iso_lang = self.language_detector.INDIC_LANG_MAPPING[detected_lang]
            
            # Check if language is allowed
            should_include = (
                iso_lang in self.allowed_languages and 
                confidence >= self.min_language_confidence
            )
            
            logger.debug(
                f"Language detection: {iso_lang} (confidence: {confidence:.2f}, "
                f"mixed: {is_mixed}, include: {should_include})"
            )
            
            return iso_lang, confidence, should_include
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            # On error, assume English and include the chunk
            return 'en', 0.5, True

    def _clean_text_for_language_detection(self, text: str) -> str:
        """
        Clean text for more accurate language detection.
        Removes markdown syntax, image placeholders, URLs, etc.
        """
        try:
            # Remove image placeholders
            text = re.sub(r'<!-- IMAGE_ID:.*?-->', '', text)
            
            # Remove markdown images
            text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove markdown formatting (bold, italic, headers)
            text = re.sub(r'[#*_`~]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Text cleaning failed: {e}, using original text")
            return text


    async def get_contextualized_chunks_async(
            self,     
            filter_by_language: bool = True,
            print_chunk: bool = True) -> List[tuple]:
        if not self.chunks:
            raise ValueError("No chunks available. Call create_chunks_async() first.")
        
        results = []
        batch_size = 10
        filtered_count = 0 
        for i in range(0, len(self.chunks), batch_size):
            batch_end = min(i + batch_size, len(self.chunks))
            batch_tasks = [
                self.analyze_chunk_async(chunk_pos)
                for chunk_pos in range(i, batch_end)
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            
            for chunk_pos, (has_image, ctx_text, num_tokens) in enumerate(batch_results, start=i):
                should_include = True
                detected_lang = 'en'
                confidence = 1.0
                
                if filter_by_language and self.language_detector:
                    detected_lang, confidence, should_include = await self._detect_chunk_language(ctx_text)
                    
                    if not should_include:
                        filtered_count += 1
                        logger.info(
                            f"Chunk {chunk_pos} filtered out: language={detected_lang}, "
                            f"confidence={confidence:.2f}, allowed={self.allowed_languages}"
                        )
                        continue  # Skip this chunk                
                
                
                results.append((self.chunks[chunk_pos], ctx_text, num_tokens))
                if print_chunk:
                    chunk = self.chunks[chunk_pos]
                    doc_items_refs = [it.self_ref for it in chunk.meta.doc_items]
                    image_indicator = " IMAGE" if has_image else ""
                    lang_info = f" LANG:{detected_lang}({confidence:.2f})" if filter_by_language else ""
                    
                    title = (
                        f"chunk_pos={chunk_pos} num_tokens={num_tokens} "
                        f"doc_items_refs={doc_items_refs}{image_indicator}{lang_info}"
                    )
                    self.console.print(Panel(ctx_text, title=title))
                    
        return results


    def _inject_images_into_chunks(
        self,
        chunk_data: List[tuple],
        image_map: Dict
    ) -> List[tuple]:
        """
        Dynamically match extracted images to chunks using provenance geometry.
        
        Works for ANY content type Docling classifies as picture — no hardcoded 
        labels, no ref matching, no content-type assumptions.

        Matching strategy:
        1. Build provenance (page + bbox) for every chunk
        2. Build provenance (page + bbox) for every image  
        3. For each image, find the chunk on the same page with the highest 
           bounding-box overlap (IoU). If no bbox available, find the chunk 
           whose bbox center is nearest to the image bbox center.
        4. If no bbox data at all, fall back to first unassigned chunk on same page.
        """
        if not image_map:
            logger.info("[IMAGE] No images to inject")
            return chunk_data

        # ── Step 1: Extract provenance for every chunk ──
        chunk_provs: List[Dict] = []
        for chunk, ctx_text, num_tokens in chunk_data:
            prov = self._extract_chunk_provenance(chunk)
            chunk_provs.append(prov)

        # ── Step 2: Extract provenance for every image ──
        image_provs: Dict[str, Dict] = {}
        for ref, img_info in image_map.items():
            image_provs[ref] = img_info.get('prov', {'page_no': img_info.get('page_number', 1), 'bbox': None})

        # ── Debug: log provenance data ──
        print(f"[IMAGE] Matching {len(image_map)} images to {len(chunk_data)} chunks using provenance")
        for ref, prov in image_provs.items():
            print(f"[DEBUG] Image {image_map[ref]['id']}: page={prov.get('page_no')}, bbox={prov.get('bbox')}")
        for idx, prov in enumerate(chunk_provs):
            if prov.get('bbox'):
                print(f"[DEBUG] Chunk[{idx}]: page={prov.get('page_no')}, bbox={prov.get('bbox')}")

        # ── Step 3: Match each image to best chunk ──
        injected_images: set = set()
        # chunk_idx -> list of (img_ref, img_info) to inject
        injection_plan: Dict[int, List[tuple]] = {}

        for img_ref, img_prov in image_provs.items():
            img_info = image_map[img_ref]
            img_page = img_prov.get('page_no', 1)
            img_bbox = img_prov.get('bbox')

            # Find all chunks on the same page
            candidate_chunks = []
            for idx, c_prov in enumerate(chunk_provs):
                if c_prov.get('page_no') == img_page:
                    candidate_chunks.append((idx, c_prov))

            if not candidate_chunks:
                logger.warning(f"[IMAGE] No chunks on page {img_page} for image {img_info['id']}")
                print(f"[WARN] No chunks on page {img_page} for image {img_info['id']}")
                continue

            best_idx = self._find_best_chunk_for_image(
                img_bbox, candidate_chunks, chunk_data, injected_images
            )

            if best_idx is not None:
                if best_idx not in injection_plan:
                    injection_plan[best_idx] = []
                injection_plan[best_idx].append((img_ref, img_info))
                injected_images.add(img_ref)

        # ── Step 4: Apply injections ──
        updated = []
        for idx, (chunk, ctx_text, num_tokens) in enumerate(chunk_data):
            if idx in injection_plan:
                for img_ref, img_info in injection_plan[idx]:
                    placeholder = (
                        f"\n<!-- IMAGE_ID:{img_info['id']}"
                        f"|PATH:{img_info['path']}"
                        f"|CAPTION:{img_info.get('caption', '')} -->"
                    )
                    ctx_text += placeholder
                    logger.info(f"[IMAGE] Injected {img_info['id']} into chunk[{idx}] (page={chunk_provs[idx].get('page_no')})")
                    print(f"[IMAGE] Injected {img_info['id']} into chunk[{idx}] (page={chunk_provs[idx].get('page_no')})")
            updated.append((chunk, ctx_text, num_tokens))

        # ── Log results ──
        matched = len(injected_images)
        total = len(image_map)
        unmatched = set(image_map.keys()) - injected_images
        logger.info(f"[IMAGE] Injection complete: {matched}/{total} images matched")
        print(f"[IMAGE] Injection complete: {matched}/{total} images matched")
        if unmatched:
            logger.warning(f"[IMAGE] {len(unmatched)} unmatched: {[image_map[r]['id'] for r in unmatched]}")
            print(f"[WARN] Unmatched images: {[image_map[r]['id'] for r in unmatched]}")

        return updated

    def _extract_chunk_provenance(self, chunk) -> Dict:
        """
        Extract page number and bounding box from a chunk's doc_items provenance.
        Merges bbox across all doc_items in the chunk (union of all bboxes).
        """
        result = {'page_no': 1, 'bbox': None}
        try:
            doc_chunk = DocChunk.model_validate(chunk)
            if not (doc_chunk.meta and doc_chunk.meta.doc_items):
                return result

            min_l, min_t, max_r, max_b = float('inf'), float('inf'), float('-inf'), float('-inf')
            has_bbox = False

            for item in doc_chunk.meta.doc_items:
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no'):
                            result['page_no'] = prov.page_no

                        bbox = getattr(prov, 'bbox', None) or getattr(prov, 'bounding_box', None)
                        if bbox is not None:
                            if hasattr(bbox, 'l'):
                                min_l = min(min_l, float(bbox.l))
                                min_t = min(min_t, float(bbox.t))
                                max_r = max(max_r, float(bbox.r))
                                max_b = max(max_b, float(bbox.b))
                                has_bbox = True
                            elif hasattr(bbox, 'x'):
                                min_l = min(min_l, float(bbox.x))
                                min_t = min(min_t, float(bbox.y))
                                max_r = max(max_r, float(bbox.x + bbox.width))
                                max_b = max(max_b, float(bbox.y + bbox.height))
                                has_bbox = True
                            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                min_l = min(min_l, float(bbox[0]))
                                min_t = min(min_t, float(bbox[1]))
                                max_r = max(max_r, float(bbox[2]))
                                max_b = max(max_b, float(bbox[3]))
                                has_bbox = True

            if has_bbox:
                result['bbox'] = {'l': min_l, 't': min_t, 'r': max_r, 'b': max_b}

        except Exception as e:
            logger.warning(f"[IMAGE] Chunk provenance extraction failed: {e}")

        return result

    def _find_best_chunk_for_image(
        self,
        img_bbox: Optional[Dict],
        candidate_chunks: List[tuple],  # [(chunk_idx, chunk_prov), ...]
        chunk_data: List[tuple],
        already_injected: set
    ) -> Optional[int]:
        """
        Find the best chunk to attach an image to, using geometric matching.
        
        Priority:
        1. Highest IoU (bounding box overlap) if both have bbox
        2. Nearest center-to-center distance if both have bbox but no overlap
        3. First chunk on page without an existing image (if no bbox available)
        """
        if not candidate_chunks:
            return None

        # If image has bbox, try geometric matching
        if img_bbox:
            scored: List[tuple] = []  # [(chunk_idx, iou, distance)]

            for chunk_idx, c_prov in candidate_chunks:
                c_bbox = c_prov.get('bbox')
                if c_bbox:
                    iou = self._compute_iou(img_bbox, c_bbox)
                    dist = self._compute_center_distance(img_bbox, c_bbox)
                    scored.append((chunk_idx, iou, dist))

            if scored:
                # First try: chunk with highest IoU (overlap)
                best_by_iou = max(scored, key=lambda x: x[1])
                if best_by_iou[1] > 0.01:  # Meaningful overlap
                    return best_by_iou[0]

                # Second try: chunk with nearest center
                best_by_dist = min(scored, key=lambda x: x[2])
                return best_by_dist[0]

        # Fallback: first chunk on page that doesn't already have an image
        for chunk_idx, _ in candidate_chunks:
            _, ctx_text, _ = chunk_data[chunk_idx]
            if "<!-- IMAGE_ID:" not in ctx_text:
                return chunk_idx

        # Last resort: first chunk on page
        return candidate_chunks[0][0]

    @staticmethod
    def _compute_iou(bbox_a: Dict, bbox_b: Dict) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        try:
            inter_l = max(bbox_a['l'], bbox_b['l'])
            inter_t = max(bbox_a['t'], bbox_b['t'])
            inter_r = min(bbox_a['r'], bbox_b['r'])
            inter_b = min(bbox_a['b'], bbox_b['b'])

            if inter_r <= inter_l or inter_b <= inter_t:
                return 0.0

            inter_area = (inter_r - inter_l) * (inter_b - inter_t)
            area_a = (bbox_a['r'] - bbox_a['l']) * (bbox_a['b'] - bbox_a['t'])
            area_b = (bbox_b['r'] - bbox_b['l']) * (bbox_b['b'] - bbox_b['t'])
            union_area = area_a + area_b - inter_area

            if union_area <= 0:
                return 0.0

            return inter_area / union_area
        except Exception:
            return 0.0

    @staticmethod
    def _compute_center_distance(bbox_a: Dict, bbox_b: Dict) -> float:
        """Compute Euclidean distance between bbox centers."""
        try:
            cx_a = (bbox_a['l'] + bbox_a['r']) / 2
            cy_a = (bbox_a['t'] + bbox_a['b']) / 2
            cx_b = (bbox_b['l'] + bbox_b['r']) / 2
            cy_b = (bbox_b['t'] + bbox_b['b']) / 2
            return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
        except Exception:
            return float('inf')

    def _inject_images_into_chunks_old(
        self,
        chunk_data: List[tuple],
        image_map: Dict
    ) -> List[tuple]:
        """
        Post-process: match real images from image_map to chunks by self_ref,
        and inject IMAGE_ID placeholders into the chunk text.
        
        Each chunk has doc_items with self_ref. If any self_ref matches a key in 
        image_map, we append the image placeholder to that chunk's text.
        """
        if not image_map:
            return chunk_data

        # Build a set of image self_refs for fast lookup
        image_refs = set(image_map.keys())

        updated = []
        for chunk, ctx_text, num_tokens in chunk_data:
            try:
                doc_chunk = DocChunk.model_validate(chunk)
                injected = False
                
                if doc_chunk.meta and doc_chunk.meta.doc_items:
                    for item in doc_chunk.meta.doc_items:
                        ref = str(item.self_ref)
                        if ref in image_refs:
                            img_info = image_map[ref]
                            placeholder = (
                                f"\n<!-- IMAGE_ID:{img_info['id']}"
                                f"|PATH:{img_info['path']}"
                                f"|CAPTION:{img_info['caption']} -->"
                            )
                            ctx_text = ctx_text + placeholder
                            injected = True
                            print(f"[IMAGE] Injected image {img_info['id']} into chunk (ref={ref})")

                if not injected:
                    # Also try matching by page number for pictures without exact ref match
                    chunk_page = self.docling_qdrant_adapter.extract_page_number(chunk)
                    for ref, img_info in image_map.items():
                        if ref.startswith("__pic_"):
                            # Unnamed picture — match by page
                            if img_info.get('page_number') == chunk_page:
                                # Check if this image was already injected into another chunk
                                if not img_info.get('_injected'):
                                    placeholder = (
                                        f"\n<!-- IMAGE_ID:{img_info['id']}"
                                        f"|PATH:{img_info['path']}"
                                        f"|CAPTION:{img_info['caption']} -->"
                                    )
                                    ctx_text = ctx_text + placeholder
                                    img_info['_injected'] = True
                                    print(f"[IMAGE] Injected image {img_info['id']} into chunk by page ({chunk_page})")
                                    break

            except Exception as e:
                logger.warning(f"Failed to check chunk for image injection: {e}")

            updated.append((chunk, ctx_text, num_tokens))

        return updated

    async def close(self):
        if self.serializer_provider:
            await self.serializer_provider.close()
        if self.picture_processor:
            await self.picture_processor.close()
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        await self.initialize_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ── Main execution function ──
    async def process_document_main(self, file_id: str, file_name: str, pdfpath: str,filter_by_language: bool = True):
        """Main async function to process document"""
        qdrant_touched = False
        try:
            await self.initialize_async()

            # Step 1: Convert PDF to Docling document
            print(" Converting document...")
            await self.convert_document_async(pdfpath)
            print(" Document converted")

            # Step 2: Extract REAL images from the document (post-processing, not serializer)
            print("[INFO] Extracting real images from document...")
            image_map = await self.picture_processor.extract_and_save_real_images(self.doc)
            print(f"[INFO] Found {len(image_map)} genuine images")

            # Step 3: Create chunks (uses default MarkdownPictureSerializer — no custom image saving)
            print("\n Creating chunks...")
            await self.create_chunks_async()
            print(f" Created {len(self.chunks)} chunks")

            # Step 4: Contextualize chunks
            print(f"[INFO] Contextualizing chunks...")
            chunk_data = await self.get_contextualized_chunks_async(filter_by_language=filter_by_language)

            # Step 5: Inject real image references into matching chunks
            print(f"[INFO] Injecting image references into chunks...")
            chunk_data = self._inject_images_into_chunks(chunk_data, image_map)

            # Step 6: Prepare for Qdrant
            chunks, texts, tokens = zip(*chunk_data)
            chunks_metadata = self.docling_qdrant_adapter.prepare_chunks_for_qdrant(
                list(chunks), list(texts), list(tokens)
            )

            # Step 7: Insert into Qdrant
            print(f"[INFO] Inserting into Qdrant...")
            qdrant_touched = True
            results = await self.qdrant_service.insert_docling_chunks(
                chunks_metadata=chunks_metadata,
                file_id=file_id,
                filename=file_name,
                check_duplicates=True
            )

            print(f"[INFO] Qdrant insertion results: {results}")

            if results > 0:
                print(f"[SUCCESS] Vector inserted successfully for {file_id}")
                return {"status": "inserted", "file_id": file_id, "count": results}
            else:
                print(f"[INFO] No new vectors for {file_id}")
                logger.error(f"No vectors inserted for {file_id}")
                return {"status": "duplicate", "file_id": file_id, "count": 0}

        except Exception as e:
            print(f"Pipeline failed for {file_name}: {e}")
            logger.error(f"Pipeline failed for {file_name}: {e}")
            if qdrant_touched:
                return {"status": "partial_failure", "file_id": file_id, "count": 0}
            else:
                return {"status": "failed", "file_id": file_id, "count": 0}  
