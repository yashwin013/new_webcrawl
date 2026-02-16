# import os
# import uuid
# import re
# import numpy as np
# from datetime import datetime
# from typing import List, Dict, Tuple, Optional
# from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
# from qdrant_client import QdrantClient, models
# from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Distance, VectorParams
# from qdrant_client.http.models import PayloadSchemaType
# from dataclasses import asdict
# import json
# from docling_core.transforms.chunker.base import BaseChunk
# # from app.bot.kbvectordb.enhanced_pdf_processor import PageChunk, DocumentMetadata
# from dataclasses import dataclass

# @dataclass
# class DocumentMetadata:
#     filename: str
#     total_pages: int
#     total_chunks: int
#     file_id: Optional[str] = None
#     page_chunks_map: Optional[Dict] = None

# @dataclass
# class PageChunk:
#     """Chunk data structure for batch processing compatibility"""
#     chunk_id: str
#     text: str
#     page_number: int
#     chunk_index: int
#     page_context: str = ""
#     previous_chunk_id: Optional[str] = None
#     next_chunk_id: Optional[str] = None
#     file_id: Optional[str] = None

# from app.schemas.vector import VectorMetaData
# from app.docling.qdrant_adapter import DoclingChunkMetadata
# from app.config import app_config
# import logging
# logger = logging.getLogger("docling_qdrant_services")

# class LocalEmbeddingWrapper:
#     """Wrapper for SentenceTransformer to match expected interface"""
#     def __init__(self, model):
#         self.model = model
    
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         embeddings = self.model.encode(texts)
#         return embeddings.tolist()
    
#     def embed_query(self, text: str) -> List[float]:
#         embedding = self.model.encode(text)
#         return embedding.tolist()

# class DoclingQdrantService:
#     """Enhanced Qdrant service with page-aware vector storage and retrieval"""
    
#     def __init__(self, url: str, apikey: str, collection_name: str, embedding_model: str):
#         try:
#             self.client = QdrantClient(url=url, api_key=apikey)
#             self.collection_name = collection_name
#             self.model_name = embedding_model
            
#             # Initialize local embedding model
#             from app.config import get_embedding_model
#             model, self.vector_size = get_embedding_model()
            
#             # Create a wrapper to match the expected interface if needed, or just use it directly
#             # The insert method expects .embed_documents()
#             self.embedding_model = LocalEmbeddingWrapper(model)
            
#             self.ensure_collection()
#         except Exception as e:
#             print(f"[ERROR] Enhanced QdrantService Initialization Failed: {e}")

#     @classmethod
#     async def from_config(cls):
#         """Initialize Enhanced QdrantService with configuration"""
#         url = app_config.QDRANT_URL
#         apikey = app_config.QDRANT_API_KEY
#         collection_name = app_config.QDRANT_COLLECTION 
#         embedding_model = app_config.EMBEDDING_MODEL
        
#         return cls(url, apikey, collection_name, embedding_model)

#     def ensure_collection(self):
#         """Create enhanced collection with additional metadata fields"""
#         try:
#             collections = self.client.get_collections().collections
#             existing_collections = {col.name for col in collections}

#             if self.collection_name not in existing_collections:
#                 self._create_collection()
#             else:
#                 # Check if vector size matches
#                 try:
#                     collection_info = self.client.get_collection(self.collection_name)
#                     # Handle both single vector and multivector config
#                     current_size = None
#                     if hasattr(collection_info.config.params.vectors, 'size'):
#                         current_size = collection_info.config.params.vectors.size
                    
#                     if current_size and current_size != self.vector_size:
#                         print(f"[WARNING] Collection '{self.collection_name}' has vector size {current_size}, expected {self.vector_size}. Recreating...")
#                         self.client.delete_collection(self.collection_name)
#                         self._create_collection()
#                     else:
#                         print(f"[INFO] Enhanced collection '{self.collection_name}' exists and is valid.")
#                 except Exception as check_error:
#                     print(f"[WARNING] Failed to check collection config: {check_error}. Assuming valid.")
#         except Exception as e:
#             print(f"[ERROR] Enhanced Collection Check/Create Failed: {e}")

#     def _create_collection(self):
#         """Helper to create collection with correct schema"""
#         print(f"[INFO] Creating enhanced collection '{self.collection_name}' with size {self.vector_size}...")
        
#         self.client.create_collection(
#             collection_name=self.collection_name,
#             vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE, on_disk=True),
#             hnsw_config=models.HnswConfigDiff(
#                 m=16,
#                 ef_construct=200,
#                 full_scan_threshold=1000,
#                 max_indexing_threads=2,
#                 on_disk=True
#             ),
#             on_disk_payload=True
#         )

#         # Create enhanced payload indexes
#         indexes = [
#             ("file_id", PayloadSchemaType.KEYWORD),
#             ("chunk_id", PayloadSchemaType.KEYWORD),
#             ("page_number", PayloadSchemaType.INTEGER),
#             ("chunk_index", PayloadSchemaType.INTEGER),
#             ("has_image", PayloadSchemaType.BOOL),
#             ("is_delete", PayloadSchemaType.BOOL),
#             ("is_active", PayloadSchemaType.BOOL),
#         ]
        
#         for field_name, field_type in indexes:
#             self.client.create_payload_index(
#                 collection_name=self.collection_name,
#                 field_name=field_name,
#                 field_schema=field_type
#             )
#         print(f"[INFO] Enhanced collection '{self.collection_name}' created successfully.")

#     async def insert_docling_chunks(self, chunks_metadata: List[DoclingChunkMetadata], file_id: str, 
#                              filename: str, 
#                              check_duplicates: bool = True, 
#                              similarity_threshold: float = 0.95) -> int:
#         """
#         Insert docling chunks into Qdrant
        
#         Args:
#             chunks_metadata: List of DoclingChunkMetadata
#             file_id: File identifier
#             filename: Name of the file
#             check_duplicates: Whether to check for and skip duplicate content
#             similarity_threshold: Threshold for considering content as duplicate
            
#         Returns:
#             Number of inserted chunks
#         """
#         results = {'inserted': 0, 'skipped': 0, 'failed': 0}
#         inserted_count = 0
#         skipped_count = 0
#         failed_count = 0
#         batch_size = 50  # Process in batches
        
#         print(f"[INFO] Starting insertion of {len(chunks_metadata)} chunks for file {file_id}")
#         print(f"[INFO] Deduplication: {'ENABLED' if check_duplicates else 'DISABLED'}")
        
#         try:
#             for i in range(0, len(chunks_metadata), batch_size):
#                 batch = chunks_metadata[i:i + batch_size]
#                 batch_points = []
                
#                 for chunk_meta in batch:
#                     try:
#                         # Validate chunk has text
#                         if not chunk_meta.text or len(chunk_meta.text.strip()) == 0:
#                             print(f"[WARNING] Skipping empty chunk {chunk_meta.chunk_id}")
#                             skipped_count += 1
#                             continue
                        
#                         # Truncate text if too long
#                         chunk_text = chunk_meta.text
#                         if len(chunk_text) > 8000:
#                             chunk_text = chunk_text[:8000] + "..."
#                             print(f"[WARNING] Truncated chunk {chunk_meta.chunk_id}")
                        
#                         # Create embedding
#                         try:
#                             vector = self.embedding_model.embed_documents([chunk_text])[0]
#                         except Exception as embed_err:
#                             print(f"[ERROR] Embedding failed for chunk {chunk_meta.chunk_id}: {embed_err}")
#                             failed_count += 1
#                             continue
                            
#                         vector = np.array(vector, dtype=np.float32)
                        
#                         # Validate vector
#                         if np.isnan(vector).any() or np.isinf(vector).any():
#                             print(f"[ERROR] Invalid vector for chunk {chunk_meta.chunk_id} (NaN or Inf values)")
#                             failed_count += 1
#                             continue
                            
#                         normalized_vector = self.normalize_vector(vector)
                        
#                         # Check duplicates if enabled
#                         if check_duplicates:
#                             similar = self._check_duplicate(normalized_vector, file_id)
#                             if similar:
#                                 results['skipped'] += 1
#                                 skipped_count += 1
#                                 print(f"[INFO] Skipped duplicate chunk {chunk_meta.chunk_id}")
#                                 continue
                        
#                         # Create point
#                         try:
#                             vector_id = str(uuid.uuid4())
#                             timestamp = datetime.utcnow().isoformat()
                            
#                             payload = {
#                                 "vector_id": vector_id,
#                                 "file_id": file_id,
#                                 "filename": filename,
#                                 "chunk_id": chunk_meta.chunk_id,
#                                 "page_number": chunk_meta.page_number,
#                                 "chunk_index": chunk_meta.chunk_index,
#                                 "original_text": chunk_text,
#                                 "has_image": chunk_meta.has_image,
#                                 "token_count": chunk_meta.token_count,
#                                 "heading_text": chunk_meta.heading_text or "",
#                                 "doc_items_refs": chunk_meta.doc_items_refs,
#                                 "timestamp": timestamp,
#                                 "is_delete": False,
#                                 "is_active": True,
#                             }
                            
#                             point = PointStruct(
#                                 id=vector_id,
#                                 vector=normalized_vector.tolist(),
#                                 payload=payload
#                             )
#                             batch_points.append(point)
#                         except Exception as point_err:
#                             print(f"[ERROR] Failed to create point for chunk {chunk_meta.chunk_id}: {point_err}")
#                             failed_count += 1
#                             continue
                        
#                     except Exception as e:
#                         print(f"[ERROR] Failed to process chunk {chunk_meta.chunk_id}: {e}")
#                         results['failed'] += 1
#                         failed_count += 1
#                         continue
                
#                 # Batch insert with error handling
#                 if batch_points:
#                     try:
#                         self.client.upsert(self.collection_name, batch_points)
#                         inserted_count += len(batch_points)
#                         results['inserted'] += len(batch_points)
#                         print(f"[INFO] Inserted batch {i//batch_size + 1}: {len(batch_points)} vectors")
#                     except Exception as batch_error:
#                         print(f"[ERROR] Failed to insert batch {i//batch_size + 1}: {batch_error}")
#                         # Try inserting one by one
#                         for j, point in enumerate(batch_points):
#                             try:
#                                 self.client.upsert(self.collection_name, [point])
#                                 inserted_count += 1
#                             except Exception as single_error:
#                                 print(f"[ERROR] Failed to insert single vector {j}: {single_error}")
#                                 failed_count += 1
#                                 continue
            
#             print(f"[SUMMARY] Total chunks: {len(chunks_metadata)} | Inserted: {inserted_count} | Skipped: {skipped_count} | Failed: {failed_count}")
#             if skipped_count > 0:
#                 print(f"[INFO] Skipped {skipped_count} duplicate chunks")
#             if failed_count > 0:
#                 print(f"[WARNING] Failed to insert {failed_count} chunks")
#             return inserted_count
            
#         except Exception as e:
#             print(f"[ERROR] Batch insertion failed: {e}")
#             return inserted_count

#     def normalize_vector(self, vectors):
#         """Normalize vectors to unit length"""
#         try:
#             if isinstance(vectors, list):
#                 vectors = np.array(vectors)
            
#             if vectors.ndim == 1:
#                 norm = np.linalg.norm(vectors)
#                 return vectors / norm if norm != 0 else vectors 
#             else:
#                 norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#                 # Avoid division by zero
#                 norms[norms == 0] = 1
#                 return vectors / norms
#         except Exception as e:
#             print(f"[ERROR] Vector Normalization Failed: {e}")
#             return vectors


#     def _validate_point_id(self, point_id: str) -> str:
#         """Validate and clean point ID for Qdrant compatibility"""
#         try:
#             # Ensure point ID is valid for Qdrant (alphanumeric, no special chars except underscore)
#             # Remove any problematic characters and limit length
#             clean_id = re.sub(r'[^a-zA-Z0-9_]', '', str(point_id))
#             # Limit length to avoid issues
#             if len(clean_id) > 32:
#                 clean_id = clean_id[:32]
#             # Ensure it's not empty
#             if not clean_id:
#                 clean_id = str(uuid.uuid4()).replace('-', '')[:16]
#             return clean_id
#         except Exception as e:
#             print(f"[ERROR] Point ID validation failed: {e}")
#             return str(uuid.uuid4()).replace('-', '')[:16]


#     def insert_document_chunks(self, chunks: List[BaseChunk], file_id: str, 
#                              filename: str, doc_metadata: DocumentMetadata, 
#                              check_duplicates: bool = True, 
#                              similarity_threshold: float = 0.95) -> int:
#         """
#         Insert all chunks from a document with batch processing and duplicate detection
        
#         Args:
#             chunks: List of PageChunk objects to insert
#             file_id: File identifier
#             filename: Name of the file
#             doc_metadata: Document metadata
#             check_duplicates: Whether to check for and skip duplicate content
#             similarity_threshold: Threshold for considering content as duplicate (0.0-1.0)
        
#         Returns:
#             Number of chunks successfully inserted
#         """
#         inserted_count = 0
#         skipped_count = 0
#         batch_size = 50  # Process in batches
        
#         try:
#             for i in range(0, len(chunks), batch_size):
#                 batch_chunks = chunks[i:i + batch_size]
#                 batch_points = []
                
#                 for chunk in batch_chunks:
#                     # Validate and truncate text if too long
#                     chunk_text = chunk.text
#                     if len(chunk_text) > 8000:  # Most embedding models have token limits
#                         chunk_text = chunk_text[:8000] + "..."
#                         print(f"[WARNING] Truncated long chunk text: {chunk.chunk_id}")
                    
#                     # Create embedding
#                     vector = self.embedding_model.embed_documents([chunk_text])[0]
#                     vector = np.array(vector, dtype=np.float32)
#                     normalized_vector = self.normalize_vector(vector)
                    
#                     # Check for duplicates using similarity search (if enabled)
#                     if check_duplicates:
#                         similar_vectors = self.search_similar_vectors(normalized_vector, top_k=1)
#                         if similar_vectors and similar_vectors[0]["score"] > similarity_threshold:
#                             print(f"[INFO] Skipping insertion: Similar content already exists for chunk {chunk.chunk_id} (similarity: {similar_vectors[0]['score']:.3f})")
#                             skipped_count += 1
#                             continue  # Skip this chunk as it's too similar to existing content
                    
#                     # Create metadata
#                     timestamp = datetime.utcnow().isoformat()
#                     # Create a short, valid UUID for Qdrant
#                     raw_id = str(uuid.uuid4()).replace('-', '')[:16]  # Short 16-char string
#                     # vector_id = self._validate_point_id(raw_id)
#                     vector_id = str(uuid.uuid4()) 
                    
#                     metadata = {
#                         "vector_id": vector_id,
#                         "file_id": file_id,
#                         "filename": filename,
#                         "timestamp": timestamp,
#                         "original_text": chunk.text,
#                         "is_delete": False,
#                         "is_active": True,
#                         "chunk_id": chunk.chunk_id,
#                         "page_number": chunk.page_number,
#                         "start_char": chunk.start_char,
#                         "end_char": chunk.end_char,
#                         "chunk_index": chunk.chunk_index,
#                         "is_cross_page": chunk.chunk_index == -1,
#                         "page_context": (chunk.page_context or "")[:500],  # Limit context size
#                         "previous_chunk_id": chunk.previous_chunk_id or "",
#                         "next_chunk_id": chunk.next_chunk_id or "",
#                         "document_title": doc_metadata.filename,
#                         "total_pages": doc_metadata.total_pages,
#                         "total_chunks": doc_metadata.total_chunks,
#                     }
                    
#                     point = PointStruct(id=vector_id, vector=normalized_vector, payload=metadata)
#                     batch_points.append(point)
                
#                 # Batch insert with error handling
#                 try:
#                     self.client.upsert(self.collection_name, batch_points)
#                     inserted_count += len(batch_points)
#                     print(f"[INFO] Inserted batch {i//batch_size + 1}: {len(batch_points)} vectors")
#                 except Exception as batch_error:
#                     print(f"[ERROR] Failed to insert batch {i//batch_size + 1}: {batch_error}")
#                     # Try inserting one by one to identify problematic vectors
#                     for j, point in enumerate(batch_points):
#                         try:
#                             self.client.upsert(self.collection_name, [point])
#                             inserted_count += 1
#                         except Exception as single_error:
#                             print(f"[ERROR] Failed to insert single vector {j} in batch: {single_error}")
#                             # Log the problematic point for debugging
#                             print(f"[DEBUG] Problematic point ID: {point.id}")
#                             continue
            
#             print(f"[INFO] Total enhanced vectors inserted: {inserted_count}")
#             if skipped_count > 0:
#                 print(f"[INFO] Skipped {skipped_count} duplicate chunks (similarity > {similarity_threshold})")
#             print(f"[INFO] Processing summary: {inserted_count} inserted, {skipped_count} skipped, {len(chunks)} total chunks")
#             return inserted_count
            
#         except Exception as e:
#             print(f"[ERROR] Batch insertion failed: {e}")
#             return inserted_count
        
    
#     def get_chunk_neighbors(self, chunk_id: str, file_id: str, 
#                            neighbor_count: int = 2) -> Dict[str, List[Dict]]:
#         """Get neighboring chunks for better context"""
#         try:
#             # Get the specific chunk first
#             query_filter = Filter(
#                 must=[
#                     FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id)),
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id)),
#                     FieldCondition(key="is_delete", match=MatchValue(value=False)),
#                     FieldCondition(key="is_active", match=MatchValue(value=True))
#                 ]
#             )
            
#             results = self.client.scroll(
#                 collection_name=self.collection_name,
#                 scroll_filter=query_filter,
#                 limit=1,
#                 with_payload=True
#             )
            
#             if not results[0]:
#                 return {"previous": [], "next": [], "same_page": []}
            
#             current_chunk = results[0][0].payload
#             page_number = current_chunk["page_number"]
#             chunk_index = current_chunk["chunk_index"]
            
#             # Get neighboring chunks on the same page
#             page_filter = Filter(
#                 must=[
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id)),
#                     FieldCondition(key="page_number", match=MatchValue(value=page_number)),
#                     FieldCondition(key="is_delete", match=MatchValue(value=False)),
#                     FieldCondition(key="is_active", match=MatchValue(value=True))
                   
#                 ]
#             )
            
#             page_results = self.client.scroll(
#                 collection_name=self.collection_name,
#                 scroll_filter=page_filter,
#                 limit=100,  # Get all chunks from the page
#                 with_payload=True
#             )
            
#             # Sort by chunk_index and get neighbors
#             page_chunks = sorted(page_results[0], key=lambda x: x.payload["chunk_index"])
            
#             current_index = None
#             for i, chunk in enumerate(page_chunks):
#                 if chunk.payload["chunk_id"] == chunk_id:
#                     current_index = i
#                     break
            
#             neighbors = {
#                 "previous": [],
#                 "next": [],
#                 "same_page": []
#             }
            
#             if current_index is not None:
#                 # Get previous chunks
#                 for i in range(max(0, current_index - neighbor_count), current_index):
#                     chunk_data = {
#                         "text": page_chunks[i].payload["original_text"],
#                         "chunk_id": page_chunks[i].payload["chunk_id"],
#                         "page_number": page_chunks[i].payload["page_number"],
#                         "chunk_index": page_chunks[i].payload["chunk_index"]
#                     }
#                     neighbors["previous"].append(chunk_data)
                
#                 # Get next chunks
#                 for i in range(current_index + 1, min(len(page_chunks), current_index + 1 + neighbor_count)):
#                     chunk_data = {
#                         "text": page_chunks[i].payload["original_text"],
#                         "chunk_id": page_chunks[i].payload["chunk_id"],
#                         "page_number": page_chunks[i].payload["page_number"],
#                         "chunk_index": page_chunks[i].payload["chunk_index"]
#                     }
#                     neighbors["next"].append(chunk_data)
                
#                 # Get all chunks from the same page for context
#                 for chunk in page_chunks:
#                     if chunk.payload["chunk_id"] != chunk_id:
#                         chunk_data = {
#                             "text": chunk.payload["original_text"],
#                             "chunk_id": chunk.payload["chunk_id"],
#                             "chunk_index": chunk.payload["chunk_index"]
#                         }
#                         neighbors["same_page"].append(chunk_data)
            
#             return neighbors
            
#         except Exception as e:
#             print(f"[ERROR] Failed to get chunk neighbors: {e}")
#             return {"previous": [], "next": [], "same_page": []}


# #######





#     def _check_duplicate(self, vector: np.ndarray, file_id: str, threshold: float = 0.95) -> bool:
#         """Check if similar vector already exists"""
#         try:
#             query_filter = Filter(
#                 must=[
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id)),
#                     FieldCondition(key="is_delete", match=MatchValue(value=False)),
#                 ]
#             )
            
#             results = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=vector.tolist(),
#                 limit=1,
#                 query_filter=query_filter
#             ).points
            
#             return bool(results and results[0].score > threshold)
#         except Exception:
#             return False

#     def delete_by_file_id(self, file_id: str) -> int:
#         """Hard delete all vectors for a file"""
#         try:
#             query_filter = Filter(
#                 must=[
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id))
#                 ]
#             )
            
#             # Count first
#             count_result = self.client.count(
#                 collection_name=self.collection_name,
#                 count_filter=query_filter,
#                 exact=True
#             )
            
#             if count_result.count > 0:
#                 self.client.delete(
#                     collection_name=self.collection_name,
#                     points_selector=models.FilterSelector(filter=query_filter)
#                 )
#                 print(f"[INFO] Deleted {count_result.count} vectors for file {file_id}")
#                 return count_result.count
            
#             return 0
            
#         except Exception as e:
#             print(f"[ERROR] Deletion failed: {e}")
#             return 0

#     def search_vectors_old(self, query: str, top_k: int = 10, file_id: Optional[str] = None) -> List[Dict]:
#         """Search for similar vectors"""
#         try:
#             query_vector = self.embedding_model.embed_query(query)
#             query_vector = np.array(query_vector, dtype=np.float32)
#             normalized_vector = self.normalize_vector(query_vector)
            
#             # Build filter
#             filter_conditions = [
#                 FieldCondition(key="is_delete", match=MatchValue(value=False)),
#                 FieldCondition(key="is_active", match=MatchValue(value=True))
#             ]
            
#             if file_id:
#                 filter_conditions.append(
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id))
#                 )
            
#             query_filter = Filter(must=filter_conditions)
            
#             results = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=normalized_vector.tolist(),
#                 limit=top_k,
#                 with_payload=True,
#                 query_filter=query_filter
#             ).points
            
#             return [
#                 {
#                     "text": r.payload["original_text"],
#                     "score": r.score,
#                     "page_number": r.payload["page_number"],
#                     "chunk_id": r.payload["chunk_id"],
#                     "has_image": r.payload.get("has_image", False),
#                     "heading": r.payload.get("heading_text", ""),
#                 }
#                 for r in results
#             ]
            
#         except Exception as e:
#             print(f"[ERROR] Search failed: {e}")
#             return []
        

#     def search_vectors(
#         self, 
#         query: str, 
#         top_k: int = 10, 
#         file_id: Optional[str] = None,
#         page_number: Optional[int] = None,
#         include_context: bool = False,
#         context_window: int = 2
#     ) -> List[Dict]:
#         """
#         Production-ready search with optional context enhancement
        
#         Args:
#             query: Search query string
#             top_k: Number of results to return
#             file_id: Optional file filter
#             page_number: Optional page filter
#             include_context: Whether to include neighboring chunks for context
#             context_window: Number of neighboring chunks to include (if include_context=True)
        
#         Returns:
#             List of search results with metadata
#         """
#         try:
#             # Create query vector
#             query_vector = self.embedding_model.embed_query(query)
#             query_vector = np.array(query_vector, dtype=np.float32)
#             normalized_vector = self.normalize_vector(query_vector)
            
#             # Build filter conditions
#             filter_conditions = [
#                 FieldCondition(key="is_delete", match=MatchValue(value=False)),
#                 FieldCondition(key="is_active", match=MatchValue(value=True))
#             ]
            
#             # Add file filter if specified
#             if file_id:
#                 filter_conditions.append(
#                     FieldCondition(key="file_id", match=MatchValue(value=file_id))
#                 )
            
#             # Add page filter if specified
#             if page_number is not None:
#                 filter_conditions.append(
#                     FieldCondition(key="page_number", match=MatchValue(value=page_number))
#                 )
            
#             query_filter = Filter(must=filter_conditions)
            
#             # Perform search
#             results = self.client.query_points(
#                 collection_name=self.collection_name,
#                 query=normalized_vector.tolist(),
#                 limit=top_k,
#                 with_payload=True,
#                 query_filter=query_filter
#             ).points
            
#             # Process results
#             search_results = []
#             for r in results:
#                 payload = r.payload
                
#                 # Base result
#                 result = {
#                     "text": payload["original_text"],
#                     "score": r.score,
#                     "page_number": payload.get("page_number", 1),
#                     "chunk_id": payload.get("chunk_id", ""),
#                     "chunk_index": payload.get("chunk_index", 0),
#                     "has_image": payload.get("has_image", False),
#                     "heading": payload.get("heading_text", ""),
#                     "file_id": payload.get("file_id", ""),
#                     "token_count": payload.get("token_count", 0),
#                     "doc_items_refs": payload.get("doc_items_refs", []),
#                 }
                
#                 # Add context if requested
#                 if include_context and payload.get("chunk_id") and payload.get("file_id"):
#                     neighbors = self.get_chunk_neighbors(
#                         chunk_id=payload["chunk_id"],
#                         file_id=payload["file_id"],
#                         neighbor_count=context_window
#                     )
                    
#                     # Build extended text with context
#                     extended_text = payload["original_text"]
#                     context_parts = []
                    
#                     # Add previous chunks
#                     if neighbors.get('previous'):
#                         prev_texts = [chunk['text'] for chunk in reversed(neighbors['previous'])]
#                         context_parts.extend(prev_texts)
                    
#                     # Add current chunk
#                     context_parts.append(extended_text)
                    
#                     # Add next chunks
#                     if neighbors.get('next'):
#                         next_texts = [chunk['text'] for chunk in neighbors['next']]
#                         context_parts.extend(next_texts)
                    
#                     # Join with context separator
#                     extended_text = "\n\n[...]\n\n".join(context_parts)
                    
#                     result.update({
#                         "text_with_context": extended_text,
#                         "original_text": payload["original_text"],
#                         "has_context": True,
#                         "context_chunks_count": len(neighbors.get('previous', [])) + len(neighbors.get('next', [])),
#                         "previous_chunks": neighbors.get('previous', []),
#                         "next_chunks": neighbors.get('next', []),
#                     })




#     async def search_vectors_async(
#         self, 
#         query: str, 
#         top_k: int = 10, 
#         file_id: Optional[str] = None,
#         page_number: Optional[int] = None,
#         include_context: bool = False,
#         context_window: int = 2
#     ) -> List[Dict]:
#         """
#         Async version of search_vectors for better concurrency
        
#         Args:
#             Same as search_vectors
        
#         Returns:
#             List of search results with metadata
#         """
#         import asyncio
        
#         return await asyncio.to_thread(
#             self.search_vectors,
#             query=query,
#             top_k=top_k,
#             file_id=file_id,
#             page_number=page_number,
#             include_context=include_context,
#             context_window=context_window
#         )           




import os
import uuid
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import asyncio
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, Distance, VectorParams
from qdrant_client.http.models import PayloadSchemaType
from dataclasses import asdict
import json
from docling_core.transforms.chunker.base import BaseChunk
# from app.bot.kbvectordb.enhanced_pdf_processor import PageChunk, DocumentMetadata
            
# from app.docling knowledgebase.schemas import VectorMetaData
from app.docling.qdrant_adapter import DoclingChunkMetadata
from app.config import app_config
from app.core.circuit_breaker import CircuitBreaker
import logging
logger = logging.getLogger("docling_qdrant_services")

class LocalEmbeddingWrapper:
    """Wrapper for SentenceTransformer to match expected interface"""
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

class DoclingQdrantService:
    """Enhanced Qdrant service with page-aware vector storage and retrieval"""
    
    def __init__(self, url: str, apikey: str, collection_name: str, embedding_model: str):
        try:
            self.client = QdrantClient(url=url, api_key=apikey)
            self.collection_name = collection_name
            self.model_name = embedding_model
            
            # Circuit breaker for Qdrant operations
            self._breaker = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=60,
                success_threshold=2
            )
            
            # Initialize local embedding model
            from app.config import get_embedding_model
            model, self.vector_size = get_embedding_model()
            
            # Wrap the model to match expected interface
            self.embedding_model = LocalEmbeddingWrapper(model)
            
            # Note: ensure_collection() is now async and must be called separately
        except Exception as e:
            print(f"[ERROR] Enhanced QdrantService Initialization Failed: {e}")

    @classmethod
    async def from_config(cls):
        """Initialize Enhanced QdrantService with configuration and ensure collection exists"""
        url = app_config.QDRANT_URL
        apikey = app_config.QDRANT_API_KEY
        collection_name = app_config.QDRANT_COLLECTION
        embedding_model = app_config.EMBEDDING_MODEL
        
        instance = cls(url, apikey, collection_name, embedding_model)
        await instance.ensure_collection()
        return instance

    async def ensure_collection(self):
        """Create enhanced collection with additional metadata fields"""
        try:
            collections = await asyncio.to_thread(
                lambda: self.client.get_collections().collections
            )
            existing_collections = {col.name for col in collections}

            if self.collection_name not in existing_collections:
                await self._create_collection()
            else:
                # Validate existing collection's vector size
                try:
                    collection_info = await asyncio.to_thread(
                        self.client.get_collection,
                        self.collection_name
                    )
                    current_size = None
                    
                    # Handle both single vector and named vectors config
                    if hasattr(collection_info.config.params.vectors, 'size'):
                        current_size = collection_info.config.params.vectors.size
                    
                    if current_size and current_size != self.vector_size:
                        print(f"[WARNING] Collection '{self.collection_name}' has vector size {current_size}, expected {self.vector_size}.")
                        print(f"[WARNING] Deleting and recreating collection...")
                        await asyncio.to_thread(
                            self.client.delete_collection,
                            self.collection_name
                        )
                        await self._create_collection()
                    else:
                        print(f"[INFO] Enhanced collection '{self.collection_name}' already exists with correct vector size.")
                except Exception as check_error:
                    print(f"[WARNING] Failed to validate collection config: {check_error}")
                    print(f"[INFO] Enhanced collection '{self.collection_name}' already exists.")
                
        except Exception as e:
            print(f"[ERROR] Enhanced Collection Check/Create Failed: {e}")

    async def _create_collection(self):
        """Helper method to create collection with proper schema"""
        print(f"[INFO] Creating enhanced collection '{self.collection_name}' with vector size {self.vector_size}...")
        
        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
                full_scan_threshold=1000,
                max_indexing_threads=2,
                on_disk=True
            ),
            on_disk_payload=True
        )

        # Create enhanced payload indexes
        indexes = [
            ("file_id", PayloadSchemaType.TEXT),
            ("chunk_id", PayloadSchemaType.TEXT),
            ("page_number", PayloadSchemaType.INTEGER),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("has_image", PayloadSchemaType.BOOL),
            ("is_delete", PayloadSchemaType.BOOL),
            ("is_active", PayloadSchemaType.BOOL),
        ]
        
        for field_name, field_type in indexes:
            await asyncio.to_thread(
                self.client.create_payload_index,
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )

        print(f"[INFO] Enhanced collection '{self.collection_name}' created successfully.")

    def normalize_vector(self, vectors):
        """Normalize vectors to unit length"""
        try:
            if vectors.ndim == 1:
                norm = np.linalg.norm(vectors)
                return vectors / norm if norm != 0 else vectors 
            else:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / norms
        except Exception as e:
            print(f"[ERROR] Vector Normalization Failed: {e}")
            return vectors
    
    def _validate_point_id(self, point_id: str) -> str:
        """Validate and clean point ID for Qdrant compatibility"""
        try:
            # Ensure point ID is valid for Qdrant (alphanumeric, no special chars except underscore)
            # Remove any problematic characters and limit length
            clean_id = re.sub(r'[^a-zA-Z0-9_]', '', str(point_id))
            # Limit length to avoid issues
            if len(clean_id) > 32:
                clean_id = clean_id[:32]
            # Ensure it's not empty
            if not clean_id:
                clean_id = str(uuid.uuid4()).replace('-', '')[:16]
            return clean_id
        except Exception as e:
            print(f"[ERROR] Point ID validation failed: {e}")
            return str(uuid.uuid4()).replace('-', '')[:16]


    # def insert_document_chunks(self, chunks: List[BaseChunk], file_id: str, 
    #                          filename: str, doc_metadata: DocumentMetadata, 
    #                          check_duplicates: bool = True, 
    #                          similarity_threshold: float = 0.95) -> int:
    #     """
    #     Insert all chunks from a document with batch processing and duplicate detection
        
    #     Args:
    #         chunks: List of PageChunk objects to insert
    #         file_id: File identifier
    #         filename: Name of the file
    #         doc_metadata: Document metadata
    #         check_duplicates: Whether to check for and skip duplicate content
    #         similarity_threshold: Threshold for considering content as duplicate (0.0-1.0)
        
    #     Returns:
    #         Number of chunks successfully inserted
    #     """
    #     inserted_count = 0
    #     skipped_count = 0
    #     batch_size = 50  # Process in batches
        
    #     try:
    #         for i in range(0, len(chunks), batch_size):
    #             batch_chunks = chunks[i:i + batch_size]
    #             batch_points = []
                
    #             for chunk in batch_chunks:
    #                 # Validate and truncate text if too long
    #                 chunk_text = chunk.text
    #                 if len(chunk_text) > 8000:  # Most embedding models have token limits
    #                     chunk_text = chunk_text[:8000] + "..."
    #                     print(f"[WARNING] Truncated long chunk text: {chunk.chunk_id}")
                    
    #                 # Create embedding
    #                 vector = self.embedding_model.embed_documents([chunk_text])[0]
    #                 vector = np.array(vector, dtype=np.float32)
    #                 normalized_vector = self.normalize_vector(vector)
                    
    #                 # Check for duplicates using similarity search (if enabled)
    #                 if check_duplicates:
    #                     similar_vectors = self.search_similar_vectors(normalized_vector, top_k=1)
    #                     if similar_vectors and similar_vectors[0]["score"] > similarity_threshold:
    #                         print(f"[INFO] Skipping insertion: Similar content already exists for chunk {chunk.chunk_id} (similarity: {similar_vectors[0]['score']:.3f})")
    #                         skipped_count += 1
    #                         continue  # Skip this chunk as it's too similar to existing content
                    
    #                 # Create metadata
    #                 timestamp = datetime.utcnow().isoformat()
    #                 # Create a short, valid UUID for Qdrant
    #                 raw_id = str(uuid.uuid4()).replace('-', '')[:16]  # Short 16-char string
    #                 # vector_id = self._validate_point_id(raw_id)
    #                 vector_id = str(uuid.uuid4()) 
                    
    #                 metadata = {
    #                     "vector_id": vector_id,
    #                     "file_id": file_id,
    #                     "filename": filename,
    #                     "timestamp": timestamp,
    #                     "original_text": chunk.text,
    #                     "is_delete": False,
    #                     "is_active": True,
    #                     "chunk_id": chunk.chunk_id,
    #                     "page_number": chunk.page_number,
    #                     "start_char": chunk.start_char,
    #                     "end_char": chunk.end_char,
    #                     "chunk_index": chunk.chunk_index,
    #                     "is_cross_page": chunk.chunk_index == -1,
    #                     "page_context": (chunk.page_context or "")[:500],  # Limit context size
    #                     "previous_chunk_id": chunk.previous_chunk_id or "",
    #                     "next_chunk_id": chunk.next_chunk_id or "",
    #                     "document_title": doc_metadata.filename,
    #                     "total_pages": doc_metadata.total_pages,
    #                     "total_chunks": doc_metadata.total_chunks,
    #                 }
                    
    #                 point = PointStruct(id=vector_id, vector=normalized_vector, payload=metadata)
    #                 batch_points.append(point)
                
    #             # Batch insert with error handling
    #             try:
    #                 self.client.upsert(self.collection_name, batch_points)
    #                 inserted_count += len(batch_points)
    #                 print(f"[INFO] Inserted batch {i//batch_size + 1}: {len(batch_points)} vectors")
    #             except Exception as batch_error:
    #                 print(f"[ERROR] Failed to insert batch {i//batch_size + 1}: {batch_error}")
    #                 # Try inserting one by one to identify problematic vectors
    #                 for j, point in enumerate(batch_points):
    #                     try:
    #                         self.client.upsert(self.collection_name, [point])
    #                         inserted_count += 1
    #                     except Exception as single_error:
    #                         print(f"[ERROR] Failed to insert single vector {j} in batch: {single_error}")
    #                         # Log the problematic point for debugging
    #                         print(f"[DEBUG] Problematic point ID: {point.id}")
    #                         continue
            
    #         print(f"[INFO] Total enhanced vectors inserted: {inserted_count}")
    #         if skipped_count > 0:
    #             print(f"[INFO] Skipped {skipped_count} duplicate chunks (similarity > {similarity_threshold})")
    #         print(f"[INFO] Processing summary: {inserted_count} inserted, {skipped_count} skipped, {len(chunks)} total chunks")
    #         return inserted_count
            
    #     except Exception as e:
    #         print(f"[ERROR] Batch insertion failed: {e}")
    #         return inserted_count
        
    
    async def get_chunk_neighbors(self, chunk_id: str, file_id: str, 
                           neighbor_count: int = 2) -> Dict[str, List[Dict]]:
        """Get neighboring chunks for better context"""
        try:
            # Get the specific chunk first
            query_filter = Filter(
                must=[
                    FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id)),
                    FieldCondition(key="file_id", match=MatchValue(value=file_id)),
                    FieldCondition(key="is_delete", match=MatchValue(value=False)),
                    FieldCondition(key="is_active", match=MatchValue(value=True))
                ]
            )
            
            results = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True
            )
            
            if not results[0]:
                return {"previous": [], "next": [], "same_page": []}
            
            current_chunk = results[0][0].payload
            page_number = current_chunk["page_number"]
            chunk_index = current_chunk["chunk_index"]
            
            # Get neighboring chunks on the same page
            page_filter = Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id)),
                    FieldCondition(key="page_number", match=MatchValue(value=page_number)),
                    FieldCondition(key="is_delete", match=MatchValue(value=False)),
                    FieldCondition(key="is_active", match=MatchValue(value=True))
                   
                ]
            )
            
            page_results = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=page_filter,
                limit=100,
                with_payload=True
            )
            
            # Sort by chunk_index and get neighbors
            page_chunks = sorted(page_results[0], key=lambda x: x.payload["chunk_index"])
            
            current_index = None
            for i, chunk in enumerate(page_chunks):
                if chunk.payload["chunk_id"] == chunk_id:
                    current_index = i
                    break
            
            neighbors = {
                "previous": [],
                "next": [],
                "same_page": []
            }
            
            if current_index is not None:
                # Get previous chunks
                for i in range(max(0, current_index - neighbor_count), current_index):
                    chunk_data = {
                        "text": page_chunks[i].payload["original_text"],
                        "chunk_id": page_chunks[i].payload["chunk_id"],
                        "page_number": page_chunks[i].payload["page_number"],
                        "chunk_index": page_chunks[i].payload["chunk_index"]
                    }
                    neighbors["previous"].append(chunk_data)
                
                # Get next chunks
                for i in range(current_index + 1, min(len(page_chunks), current_index + 1 + neighbor_count)):
                    chunk_data = {
                        "text": page_chunks[i].payload["original_text"],
                        "chunk_id": page_chunks[i].payload["chunk_id"],
                        "page_number": page_chunks[i].payload["page_number"],
                        "chunk_index": page_chunks[i].payload["chunk_index"]
                    }
                    neighbors["next"].append(chunk_data)
                
                # Get all chunks from the same page for context
                for chunk in page_chunks:
                    if chunk.payload["chunk_id"] != chunk_id:
                        chunk_data = {
                            "text": chunk.payload["original_text"],
                            "chunk_id": chunk.payload["chunk_id"],
                            "chunk_index": chunk.payload["chunk_index"]
                        }
                        neighbors["same_page"].append(chunk_data)
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get chunk neighbors: {e}", exc_info=True)
            return {"previous": [], "next": [], "same_page": []}


#######


    async def insert_docling_chunks(
        self, 
        chunks_metadata: List[DoclingChunkMetadata],
        file_id: str,
        filename: str,
        check_duplicates: bool = True,
        batch_size: int = 50,
        similarity_threshold: float = 0.95
    ) ->int:# Dict[str, int]:
        """
        Insert docling chunks into Qdrant
        
        Returns:
            Dict with counts: {'inserted': X, 'skipped': Y, 'failed': Z}
        """
        results = {'inserted': 0, 'skipped': 0, 'failed': 0}
        inserted_count = 0
        skipped_count = 0
        batch_size = 50  # Process in batches
        try:
            for i in range(0, len(chunks_metadata), batch_size):
                batch = chunks_metadata[i:i + batch_size]
                batch_points = []
                
                for chunk_meta in batch:
                    try:
                        # Truncate text if too long
                        chunk_text = chunk_meta.text
                        if len(chunk_text) > 8000:
                            chunk_text = chunk_text[:8000] + "..."
                            print(f"[WARNING] Truncated chunk {chunk_meta.chunk_id}")
                        
                        # Create embedding (blocking operation)
                        vector = await asyncio.to_thread(
                            lambda: self.embedding_model.embed_documents([chunk_text])[0]
                        )
                        vector = np.array(vector, dtype=np.float32)
                        normalized_vector = self.normalize_vector(vector)
                        
                        # Check duplicates if enabled
                        if check_duplicates:
                            similar = await self._check_duplicate(normalized_vector, file_id)
                            if similar:
                                results['skipped'] += 1
                                skipped_count += 1
                                continue
                        
                        # Create point
                        vector_id = str(uuid.uuid4())
                        timestamp = datetime.utcnow().isoformat()
                        
                        payload = {
                            "vector_id": vector_id,
                            "file_id": file_id,
                            "filename": filename,
                            "chunk_id": chunk_meta.chunk_id,
                            "page_number": chunk_meta.page_number,
                            "chunk_index": chunk_meta.chunk_index,
                            "original_text": chunk_text,
                            "has_image": chunk_meta.has_image,
                            "token_count": chunk_meta.token_count,
                            "heading_text": chunk_meta.heading_text or "",
                            "doc_items_refs": chunk_meta.doc_items_refs,
                            "timestamp": timestamp,
                            "is_delete": False,
                            "is_active": True,
                        }
                        
                        point = PointStruct(
                            id=vector_id,
                            vector=normalized_vector.tolist(),
                            payload=payload
                        )
                        batch_points.append(point)
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process chunk {chunk_meta.chunk_id}: {e}")
                        results['failed'] += 1
                        continue
                
                # Batch insert with error handling and circuit breaker protection
                try:
                    async with self._breaker:
                        await asyncio.to_thread(
                            self.client.upsert,
                            self.collection_name,
                            batch_points
                        )
                    inserted_count += len(batch_points)
                    results['inserted'] += len(batch_points)
                    print(f"[INFO] Inserted batch {i//batch_size + 1}: {len(batch_points)} vectors")
                except Exception as batch_error:
                    print(f"[ERROR] Failed to insert batch {i//batch_size + 1}: {batch_error}")
                    # Try inserting one by one to identify problematic vectors
                    for j, point in enumerate(batch_points):
                        try:
                            await asyncio.to_thread(
                                self.client.upsert,
                                self.collection_name,
                                [point]
                            )
                            inserted_count += 1
                        except Exception as single_error:
                            print(f"[ERROR] Failed to insert single vector {j} in batch: {single_error}")
                            # Log the problematic point for debugging
                            print(f"[DEBUG] Problematic point ID: {point.id}")
                            continue
            
            print(f"[INFO] Total enhanced vectors inserted: {inserted_count}")
            if skipped_count > 0:
                print(f"[INFO] Skipped {skipped_count} duplicate chunks (similarity > {similarity_threshold})")
            print(f"[INFO] Processing summary: {inserted_count} inserted, {skipped_count} skipped, {len(chunks_metadata)} total chunks")
            return inserted_count
            
        except Exception as e:
            print(f"[ERROR] Batch insertion failed: {e}")
            return inserted_count

    async def _check_duplicate(self, vector: np.ndarray, file_id: str, threshold: float = 0.95) -> bool:
        """Check if similar vector already exists"""
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(key="is_delete", match=MatchValue(value=False)),
                    FieldCondition(key="is_active", match=MatchValue(value=True))
                ]
            )            
            
            results = await asyncio.to_thread(
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector.tolist(),
                    limit=1,
                    query_filter=query_filter
                ).points
            )
            
            return bool(results and results[0].score > threshold)
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return False

    async def delete_by_file_id(self, file_id: str) -> int:
        """Hard delete all vectors for a file"""
        try:
            query_filter = Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                ]
            )
            
            # Count first
            count_result = await asyncio.to_thread(
                self.client.count,
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True
            )
            
            if count_result.count > 0:
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=query_filter)
                )
                print(f"[INFO] Deleted {count_result.count} vectors for file {file_id}")
                return count_result.count
            
            return 0
            
        except Exception as e:
            logger.error(f"Deletion failed for file {file_id}: {e}", exc_info=True)
            return 0

    async def search_vectors_old(self, query: str, top_k: int = 10, file_id: Optional[str] = None) -> List[Dict]:
        """Search for similar vectors"""
        try:
            query_vector = await asyncio.to_thread(
                self.embedding_model.embed_query,
                query
            )
            query_vector = np.array(query_vector, dtype=np.float32)
            normalized_vector = self.normalize_vector(query_vector)
            
            # Build filter
            filter_conditions = [
                FieldCondition(key="is_delete", match=MatchValue(value=False)),
                FieldCondition(key="is_active", match=MatchValue(value=True))
            ]
            
            if file_id:
                filter_conditions.append(
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                )
            
            query_filter = Filter(must=filter_conditions)
            
            results = await asyncio.to_thread(
                lambda: self.client.query_points(
                    collection_name=self.collection_name,
                    query=normalized_vector.tolist(),
                    limit=top_k,
                    with_payload=True,
                    query_filter=query_filter
                ).points
            )
            
            return [
                {
                    "text": r.payload["original_text"],
                    "score": r.score,
                    "page_number": r.payload["page_number"],
                    "chunk_id": r.payload["chunk_id"],
                    "has_image": r.payload.get("has_image", False),
                    "heading": r.payload.get("heading_text", ""),
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
        

    async def search_vectors(
        self, 
        query: str, 
        top_k: int = 10, 
        file_id: Optional[str] = None,
        page_number: Optional[int] = None,
        include_context: bool = False,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Production-ready search with optional context enhancement
        
        Args:
            query: Search query string
            top_k: Number of results to return
            file_id: Optional file filter
            page_number: Optional page filter
            include_context: Whether to include neighboring chunks for context
            context_window: Number of neighboring chunks to include (if include_context=True)
        
        Returns:
            List of search results with metadata
        """
        try:
            # Create query vector (blocking embedding operation)
            query_vector = await asyncio.to_thread(
                self.embedding_model.embed_query,
                query
            )
            query_vector = np.array(query_vector, dtype=np.float32)
            normalized_vector = self.normalize_vector(query_vector)
            
            # Build filter conditions
            filter_conditions = [
                FieldCondition(key="is_delete", match=MatchValue(value=False)),
                FieldCondition(key="is_active", match=MatchValue(value=True))
            ]
            
            # Add file filter if specified
            if file_id:
                filter_conditions.append(
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                )
            
            # Add page filter if specified
            if page_number is not None:
                filter_conditions.append(
                    FieldCondition(key="page_number", match=MatchValue(value=page_number))
                )
            
            query_filter = Filter(must=filter_conditions)
            
            # Perform search with circuit breaker protection (blocking I/O operation)
            async with self._breaker:
                results = await asyncio.to_thread(
                    lambda: self.client.query_points(
                        collection_name=self.collection_name,
                        query=normalized_vector.tolist(),
                        limit=top_k,
                        with_payload=True,
                        query_filter=query_filter
                    ).points
                )
            
            # Process results
            search_results = []
            for r in results:
                payload = r.payload
                
                # Base result
                result = {
                    "text": payload["original_text"],
                    "score": r.score,
                    "page_number": payload.get("page_number", 1),
                    "chunk_id": payload.get("chunk_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "has_image": payload.get("has_image", False),
                    "heading": payload.get("heading_text", ""),
                    "file_id": payload.get("file_id", ""),
                    "token_count": payload.get("token_count", 0),
                    "doc_items_refs": payload.get("doc_items_refs", []),
                }
                
                # Add context if requested
                if include_context and payload.get("chunk_id") and payload.get("file_id"):
                    neighbors = await self.get_chunk_neighbors(
                        chunk_id=payload["chunk_id"],
                        file_id=payload["file_id"],
                        neighbor_count=context_window
                    )
                    
                    # Build extended text with context
                    extended_text = payload["original_text"]
                    context_parts = []
                    
                    # Add previous chunks
                    if neighbors.get('previous'):
                        prev_texts = [chunk['text'] for chunk in reversed(neighbors['previous'])]
                        context_parts.extend(prev_texts)
                    
                    # Add current chunk
                    context_parts.append(extended_text)
                    
                    # Add next chunks
                    if neighbors.get('next'):
                        next_texts = [chunk['text'] for chunk in neighbors['next']]
                        context_parts.extend(next_texts)
                    
                    # Join with context separator
                    extended_text = "\n\n[...]\n\n".join(context_parts)
                    
                    result.update({
                        "text_with_context": extended_text,
                        "original_text": payload["original_text"],
                        "has_context": True,
                        "context_chunks_count": len(neighbors.get('previous', [])) + len(neighbors.get('next', [])),
                        "previous_chunks": neighbors.get('previous', []),
                        "next_chunks": neighbors.get('next', []),
                    })
                else:
                    result["has_context"] = False
                
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
            return []     


    async def search_vectors_async(
        self, 
        query: str, 
        top_k: int = 10, 
        file_id: Optional[str] = None,
        page_number: Optional[int] = None,
        include_context: bool = False,
        context_window: int = 2
    ) -> List[Dict]:
        """
        Async version of search_vectors for better concurrency
        
        Args:
            Same as search_vectors
        
        Returns:
            List of search results with metadata
        """
        # search_vectors is already async, just call it directly
        return await self.search_vectors(
            query=query,
            top_k=top_k,
            file_id=file_id,
            page_number=page_number,
            include_context=include_context,
            context_window=context_window
        )      