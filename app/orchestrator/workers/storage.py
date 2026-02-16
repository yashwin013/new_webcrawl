"""
Storage Worker

Stores processed chunks to MongoDB and Qdrant.
CPU-intensive worker for database operations.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import StorageTask
from app.services.document_store import DocumentStore
from app.schemas.document import CrawledDocument, DocumentStatus

logger = get_logger(__name__)


class StorageWorker(BaseWorker):
    """
    Storage worker - handles database storage operations.
    
    Responsibilities (CPU):
    - Pull StorageTask from storage_queue
    - Store chunks to MongoDB
    - Update document status
    - Generate embeddings (optional)
    - Store vectors to Qdrant (optional)
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        store_to_qdrant: bool = False,
    ):
        super().__init__(worker_id, queue_manager)
        self.store_to_qdrant = store_to_qdrant
        self.document_store = DocumentStore.from_config()
        
        # Storage statistics
        self.total_chunks_stored = 0
        self.total_documents_updated = 0
    
    @property
    def worker_type(self) -> str:
        return "storage"
    
    async def get_next_task(self) -> Optional[StorageTask]:
        """Get next storage task from queue."""
        return await self.queue_manager.get_storage_task(timeout=1.0)
    
    async def process_task(self, task: StorageTask) -> bool:
        """
        Store chunks to MongoDB and optionally to Qdrant.
        
        Args:
            task: StorageTask with chunks to store
            
        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)
        
        logger.debug(
            f"[{self.worker_id}] Storing {len(task.chunks)} chunks "
            f"for {task.website_url}"
        )
        
        try:
            # Step 1: Check if document exists
            doc = await asyncio.to_thread(
                self.document_store.get_by_source_url,
                task.document_metadata.get("url", "")
            )
            
            if not doc:
                # Create new document record
                doc = await asyncio.to_thread(
                    self.document_store.create_document,
                    original_file=task.document_metadata.get("url", "unknown"),
                    source_url=task.document_metadata.get("url", ""),
                    file_path="",  # Web page, no file
                    crawl_session_id=task.crawl_session_id,
                    crawl_depth=task.document_metadata.get("depth", 0),
                    status=DocumentStatus.PROCESSING,
                )
                logger.debug(f"[{self.worker_id}] Created document: {doc.file_id}")
            
            # Step 2: Update document with processing info
            update_data = {
                "is_crawled": "1",
                "crawl_completed_at": datetime.utcnow(),
                "total_pages": 1,  # This is one page
                "pages_with_text": 1 if task.chunks else 0,
            }
            
            # Check if OCR was used
            ocr_completed = task.document_metadata.get("ocr_completed", False)
            if ocr_completed:
                update_data.update({
                    "is_ocr_required": "1",
                    "is_ocr_completed": "1",
                    "ocr_completed_at": datetime.utcnow(),
                    "pages_needing_ocr": 1,
                })
            else:
                update_data.update({
                    "is_ocr_required": "0",
                    "is_ocr_completed": "0",
                })
            
            # Add worker tracking
            update_data["processed_by_worker"] = self.worker_id
            
            # Update document
            await asyncio.to_thread(self.document_store.update_document, doc.file_id, update_data)
            
            # Step 3: Store chunks to Qdrant with embeddings
            if task.chunks:
                from app.config import get_embedding_model, get_qdrant_client
                from qdrant_client.models import PointStruct, Distance, VectorParams
                import hashlib
                
                # Get embedding model (returns tuple: model, vector_size)
                embedding_model, vector_size = get_embedding_model()
                
                # Initialize Qdrant client with proper authentication
                qdrant_client = get_qdrant_client()
                collection_name = "crawled_documents"
                
                # Ensure collection exists
                try:
                    qdrant_client.get_collection(collection_name)
                except Exception as e:
                    # Create collection with proper vector dimensions
                    try:
                        qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                        )
                        logger.info(f"[{self.worker_id}] Created Qdrant collection: {collection_name}")
                    except Exception as create_error:
                        # Collection might have been created by another worker (race condition)
                        if "already exists" in str(create_error).lower():
                            logger.debug(f"[{self.worker_id}] Collection already exists (created by another worker)")
                        else:
                            raise
                
                # Generate embeddings and store in Qdrant
                points = []
                for i, chunk in enumerate(task.chunks):
                    # Generate embedding (chunks are dicts, not objects)
                    chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk.text
                    embedding = embedding_model.encode(chunk_text).tolist()
                    
                    # Create unique ID for chunk
                    chunk_id = hashlib.md5(
                        f"{doc.file_id}_{i}_{chunk_text[:100]}".encode()
                    ).hexdigest()
                    
                    # Create point with payload (handle dict chunks)
                    chunk_metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
                    word_count = chunk.get("word_count", 0) if isinstance(chunk, dict) else getattr(chunk, "word_count", 0)
                    
                    point = PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "file_id": doc.file_id,
                            "chunk_index": i,
                            "text": chunk_text,
                            "word_count": word_count,
                            "source_url": task.document_metadata.get("url", ""),
                            "crawl_session_id": task.crawl_session_id,
                            "created_at": datetime.utcnow().isoformat(),
                            "metadata": chunk_metadata,
                        }
                    )
                    points.append(point)
                
                # Batch upsert to Qdrant
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                
                logger.info(
                    f"[{self.worker_id}] Stored {len(task.chunks)} chunks "
                    f"to Qdrant for {task.document_metadata.get('url', 'unknown')}"
                )
            
            # Step 4: Mark document as vectorized
            if task.chunks:
                await asyncio.to_thread(
                    self.document_store.update_document,
                    doc.file_id,
                    {
                        "is_vectorized": "1",
                        "vector_count": len(task.chunks),
                        "vectorization_completed_at": datetime.utcnow(),
                        "status": DocumentStatus.VECTORIZED,
                    }
                )
            else:
                # No chunks to vectorize
                await asyncio.to_thread(
                    self.document_store.update_document,
                    doc.file_id,
                    {"status": DocumentStatus.STORED}
                )
            
            # Step 5: Store in nested websites collection structure
            try:
                from app.schemas.document import PageDocument
                from urllib.parse import urlparse
                
                source_url = task.document_metadata.get("url", "")
                parsed_url = urlparse(source_url)
                website_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                
                # Create PageDocument for this HTML page
                page_doc = PageDocument(
                    file_id=doc.file_id,
                    original_file=source_url,
                    source_url=source_url,
                    file_path="",  # HTML pages don't have file paths
                    document_type="html",
                    status=DocumentStatus.VECTORIZED if task.chunks else DocumentStatus.STORED,
                    vector_count=len(task.chunks),
                    file_size=0,
                    page_count=1,
                    mime_type="text/html",
                    crawl_session_id=task.crawl_session_id,
                    crawl_depth=task.document_metadata.get("depth", 0),
                    is_deleted=False,
                    is_crawled="1",
                    is_vectorized="1" if task.chunks else "0",
                    is_ocr_required="1" if ocr_completed else "0",
                    is_ocr_completed="1" if ocr_completed else "0",
                    total_pages=1,
                    pages_with_text=1 if task.chunks else 0,
                    pages_needing_ocr=1 if ocr_completed else 0,
                )
                
                # Add page to nested websites structure
                await asyncio.to_thread(
                    self.document_store.add_page_to_website,
                    website_url=website_url,
                    crawl_session_id=task.crawl_session_id,
                    visited_url=source_url,
                    crawl_depth=task.document_metadata.get("depth", 0),
                    page_document=page_doc
                )
                
                logger.info(
                    f"[{self.worker_id}] Added HTML page to nested websites structure: "
                    f"{source_url} in {website_url}"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.worker_id}] Failed to add page to websites structure: {e}",
                    exc_info=True
                )
            
            # Update statistics
            self.total_chunks_stored += len(task.chunks)
            self.total_documents_updated += 1
            
            # Mark task complete
            task.mark_completed(
                chunks_count=len(task.chunks),
                vectors_count=len(task.chunks) if self.store_to_qdrant else 0
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to store chunks: {e}",
                exc_info=True
            )
            
            # Move to dead letter queue if max retries exceeded
            if task.retry_count >= task.max_retries:
                await self.queue_manager.put_dead_letter(
                    task,
                    f"Storage failed: {e}"
                )
            else:
                # Requeue for retry
                task.mark_failed(str(e))
                await self.queue_manager.put_storage_task(task)
                logger.debug(
                    f"[{self.worker_id}] Requeued storage task "
                    f"(retry {task.retry_count}/{task.max_retries})"
                )
            
            return False
    
    @property
    def stats(self) -> dict:
        """Get storage worker statistics."""
        base_stats = super().stats
        
        base_stats.update({
            "total_chunks_stored": self.total_chunks_stored,
            "total_documents_updated": self.total_documents_updated,
            "avg_chunks_per_document": (
                self.total_chunks_stored / self.total_documents_updated
                if self.total_documents_updated > 0 else 0
            ),
        })
        
        return base_stats
