"""Adapter to convert Docling BaseChunk to Qdrant-compatible format"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
import uuid


@dataclass
class DoclingChunkMetadata:
    """Metadata for docling chunks compatible with Qdrant"""
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    doc_items_refs: List[str]
    has_image: bool
    token_count: int
    heading_text: Optional[str] = None
    context_text: Optional[str] = None


class DoclingQdrantAdapter:
    """Adapter to bridge Docling chunks with Qdrant storage"""
    
    def __init__(self):
        pass

    @classmethod
    async def from_config(cls):
        """Initialize Enhanced QdrantService with configuration"""
        
        return cls()

    @staticmethod
    def extract_page_number(chunk: BaseChunk) -> int:
        """Extract page number from chunk metadata"""
        try:
            doc_chunk = DocChunk.model_validate(chunk)
            # Try to get page number from doc_items
            if doc_chunk.meta and doc_chunk.meta.doc_items:
                for item in doc_chunk.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        # Extract page from provenance
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                return prov.page_no
            return 1  # Default to page 1 if not found
        except Exception as e:
            print(f"[WARNING] Could not extract page number: {e}")
            return 1
    
    @staticmethod
    def extract_doc_items_refs(chunk: BaseChunk) -> List[str]:
        """Extract document item references"""
        try:
            doc_chunk = DocChunk.model_validate(chunk)
            if doc_chunk.meta and doc_chunk.meta.doc_items:
                return [str(item.self_ref) for item in doc_chunk.meta.doc_items]
            return []
        except Exception:
            return []
    
    @staticmethod
    def check_has_image(text: str) -> bool:
        """Check if chunk contains image markdown"""
        return "<!-- IMAGE_ID:" in text
        #return "![" in text and "](" in text
    
    @staticmethod
    def extract_heading(chunk: BaseChunk) -> Optional[str]:
        """Extract heading if present in chunk"""
        try:
            doc_chunk = DocChunk.model_validate(chunk)
            if doc_chunk.meta and doc_chunk.meta.headings:
                return " > ".join(doc_chunk.meta.headings)
            return None
        except Exception:
            return None
    
    @classmethod
    def convert_basechunk_to_metadata(
        cls, 
        chunk: BaseChunk, 
        chunk_index: int,
        contextualized_text: str,
        token_count: int
    ) -> DoclingChunkMetadata:
        """Convert BaseChunk to DoclingChunkMetadata"""
        
        chunk_id = f"doc_chunk_{chunk_index:04d}_{str(uuid.uuid4())[:8]}"
        page_number = cls.extract_page_number(chunk)
        doc_items_refs = cls.extract_doc_items_refs(chunk)
        has_image = cls.check_has_image(contextualized_text)
        heading_text = cls.extract_heading(chunk)
        
        return DoclingChunkMetadata(
            chunk_id=chunk_id,
            text=contextualized_text,
            page_number=page_number,
            chunk_index=chunk_index,
            doc_items_refs=doc_items_refs,
            has_image=has_image,
            token_count=token_count,
            heading_text=heading_text,
            context_text=contextualized_text[:500]  # First 500 chars as context
        )
    
    @classmethod
    def prepare_chunks_for_qdrant(
        cls,
        chunks: List[BaseChunk],
        contextualized_texts: List[str],
        token_counts: List[int]
    ) -> List[DoclingChunkMetadata]:
        """Prepare all chunks for Qdrant insertion"""
        
        if len(chunks) != len(contextualized_texts) or len(chunks) != len(token_counts):
            raise ValueError("Chunks, texts, and token counts must have same length")
        
        metadata_list = []
        for idx, (chunk, text, tokens) in enumerate(zip(chunks, contextualized_texts, token_counts)):
            metadata = cls.convert_basechunk_to_metadata(chunk, idx, text, tokens)
            metadata_list.append(metadata)
        
        return metadata_list