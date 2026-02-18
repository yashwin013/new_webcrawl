import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pathlib import Path

#from app.bot.processdocling.pictureserializer import FilePictureSerializer_new
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from docling.chunking import HybridChunker

from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer,MarkdownListSerializer,MarkdownPictureSerializer,MarkdownInlineSerializer,MarkdownTextSerializer,MarkdownParams,MarkdownKeyValueSerializer,MarkdownFormSerializer,MarkdownMetaSerializer,MarkdownAnnotationSerializer

from transformers import AutoTokenizer

from typing import Iterable, Optional

from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc.labels import DocItemLabel
from rich.console import Console
from rich.panel import Panel


from docling_surya import SuryaOcrOptions
from pathlib import Path
from app.config import app_config
class ImgPlaceholderSerializerProvider(ChunkingSerializerProvider):
    """Async-aware serializer provider for chunking with image support"""
    
    def __init__(
        self, 
        output_dir: Path = Path("exported_images"),
        executor: Optional[ThreadPoolExecutor] = None,
        #picture_serializer: Optional[FilePictureSerializer_new] = None
    ):
        self.output_dir = output_dir
        self.executor = executor
       # self._picture_serializer = picture_serializer
        self._lock = asyncio.Lock()
    
    # @property
    # def picture_serializer(self):
    #     """Lazy initialization of picture serializer"""
    #     if self._picture_serializer is None:
    #         self._picture_serializer = FilePictureSerializer_new(
    #             output_dir=self.output_dir,
    #             executor=self.executor
    #         )
    #     return self._picture_serializer

    @classmethod
    async def from_config(cls,_executor: Optional[ThreadPoolExecutor] = None):
        """Initialize Enhanced DocumentProcessor with all required dependencies"""
        #filepicture_serializer_service = await FilePictureSerializer_new.from_config()
        output_dir=output_dir=app_config.UPLOAD_DIR + app_config.exported_images
        
        return cls(output_dir=Path(output_dir),executor=_executor)#, picture_serializer=filepicture_serializer_service

    def get_serializer(self, doc):
        """Get serializer with all markdown components"""
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=MarkdownPictureSerializer(),
            # picture_serializer=self._picture_serializer,
            text_serializer=MarkdownTextSerializer(),
            table_serializer=MarkdownTableSerializer(),
            list_serializer=MarkdownListSerializer(),
            inline_serializer=MarkdownInlineSerializer(),
            key_value_serializer=MarkdownKeyValueSerializer(),
            form_serializer=MarkdownFormSerializer(),
            meta_serializer=MarkdownMetaSerializer(),
            annotation_serializer=MarkdownAnnotationSerializer()
        )
    
    async def get_serializer_async(self, doc):
        """Async version of get_serializer for async context"""
        return self.get_serializer(doc)
    
    async def close(self):
        """Cleanup method to close picture serializer and executor"""
        if self._picture_serializer:
            await self._picture_serializer.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def __enter__(self):
        """Sync context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        if self._picture_serializer:
            try:
                asyncio.run(self._picture_serializer.close())
            except RuntimeError:
                # Event loop already running
                pass