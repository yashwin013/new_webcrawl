import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional

from bson import ObjectId
from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.core.database import db_manager, get_file_collection as get_files_collection
from app.docling.qdrant_service import DoclingQdrantService
from app.docling.processor import AsyncDocumentProcessor
from app.config import app_config

logger = logging.getLogger("vector_processor")

# ============== Concurrency Control ==============
# Global semaphore to limit concurrent document processing (prevent GPU OOM)
# Initialize with config value or default to 1
_processing_semaphore = None

def _get_semaphore():
    """Get or create the processing semaphore based on config"""
    global _processing_semaphore
    if _processing_semaphore is None:
        max_concurrent = getattr(app_config, 'MAX_CONCURRENT_PROCESSING', 1)
        _processing_semaphore = asyncio.Semaphore(max_concurrent)
    return _processing_semaphore

async def get_file_collection():
    """Get the files collection from MongoDB with proper connection management."""
    return await get_files_collection()

# ============== Models ==============
class Files(BaseModel):
    id: str = Field(..., alias="_id")
    fileId: str
    originalfile: str
    createdBy: str
    isVectorized: str = "0"
    isDeleted: bool = False
    
    @classmethod
    def model_validate(cls, data):
        # Handle ObjectId conversion to string if needed
        if "_id" in data and isinstance(data["_id"], ObjectId):
            data["_id"] = str(data["_id"])
        return super().model_validate(data)

# ============== Pipeline Functions ==============

async def create_vector_pipeline(createdby: str):
    """
    Initiate vector db pipeline creating with concurrency control
    :return:
    """
    # Acquire semaphore to limit concurrent processing (prevent GPU OOM)
    semaphore = _get_semaphore()
    async with semaphore:
        _documentProcessor = await AsyncDocumentProcessor.from_config()
        
        try:
            files = await get_file_for_vector(createdby)
        except HTTPException:
            logger.info(f"No pending files found for user {createdby}")
            return

        if files:
            for file_doc in files:
                file_path = os.path.join(app_config.UPLOAD_DIR, file_doc.fileId) 
                file_name = file_doc.originalfile
                file_id = file_doc.fileId
                file_obj_id = file_doc.id

                logger.info(f"Processing file: {file_name} ({file_id})")

                # Check if file exists
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}. Marking as failed.")
                    await update_vector_file_status(file_obj_id, file_id, createdby, status="-1")
                    continue

                try:
                    processfile = await _documentProcessor.process_document_main(file_id, file_name, file_path)
                except Exception as e:
                    logger.error(f"Error processing document {file_id}: {e}", exc_info=True)
                    processfile = False

                if not processfile:
                    logger.error(f"No vectors inserted for {file_path}. Marking as failed.")
                    await update_vector_file_status(file_obj_id, file_id, createdby, status="-1")
                    continue

                try:
                    count = await update_vector_file_status(file_obj_id, file_id, createdby, status="1")
                    if count <= 0:
                        logger.error(f"DB not updated for {file_id}. Skipping file.")
                        continue
                    print(f"Vector inserted successfully for {file_id}")
                except Exception as e:
                    logger.error(f"Failed to update status for {file_id}: {e}")
        else:
            print(f"No file to process")


async def delete_file_vector_pipeline(fileid: str):
    """
    Initiate vector db pipeline delete the vector
    :return:
    """
    try:
        _qdrantService = await DoclingQdrantService.from_config()
        result = await _qdrantService.delete_by_file_id(fileid)
        
        if result > 0:
            print(f"Permanent delete the vector of {fileid}")
        else:
            logger.error(f"Error when deleting vector of file {fileid} from vector db. It might not exist.", exc_info=True)
            
    except Exception as e:
        logger.error(f"Exception in delete_file_vector_pipeline: {e}", exc_info=True)


async def update_vector_file_status(id: str, file_id: str, createdby: str, status: str = "1"):
    file_collection = await get_file_collection()
    
    # Get documents collection from db_manager
    documents_collection = db_manager.get_collection("documents")
    
    try:
        # 1. Update files collection
        update_result = await file_collection.update_one(
            {"_id": ObjectId(id), "fileId": file_id, "isDeleted": False, "createdBy": createdby},
            {"$set": {"isVectorized": status, "updatedBy": createdby, "updatedDate": str(datetime.now(timezone.utc))}}
        )
        
        # 2. Update documents collection (sync isVectorized field)
        # Map status code to human readable status for documents collection
        doc_status = "vectorized" if status == "1" else "failed" if status == "-1" else "pending"
        
        await documents_collection.update_one(
            {"fileId": file_id},
            {"$set": {
                "isVectorized": status, 
                "status": doc_status, # Also keep status in sync
                "updatedBy": createdby, 
                "updatedAt": datetime.now(timezone.utc)
            }}
        )

        if update_result.modified_count > 0:
            return update_result.modified_count
        
        raise HTTPException(status_code=404, detail="File not found or no changes made")

    except HTTPException as e:
        raise e  # Return the HTTPException error directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"when delete the file Unexpected error: {str(e)}")


async def get_file_for_vector(createdby: str) -> List[Files]:
    file_collection = await get_file_collection()
    try:
        file_cursor = file_collection.find({"isDeleted": False, "isVectorized": "0", "createdBy": createdby})
        files = await file_cursor.to_list(length=None)
        if not files:
            raise HTTPException(status_code=404, detail="File not found")

        files_data = [Files.model_validate(f) for f in files]
        return files_data

    except HTTPException as e:
        raise HTTPException(status_code=404, detail="File not found")  # Return the HTTPException error directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"when get the files for vector Unexpected error: {str(e)}")


# ============== Crawl-to-Vectorization Pipeline ==============

async def vectorize_crawled_pdfs(crawl_session_id: str) -> dict:
    """
    Vectorize all PDFs from a completed crawl session using the Process_Docling pipeline.

    Queries the 'websites' collection for PDFs with isVectorized="0",
    then calls process_document_main() which handles:
    - Docling conversion (GPU)
    - HybridChunker (automatic chunking)
    - Language filtering
    - Image extraction & injection
    - Qdrant insertion with duplicate detection

    Args:
        crawl_session_id: The crawl session ID to process PDFs for

    Returns:
        Dict with processing results summary
    """
    semaphore = _get_semaphore()
    
    results = {
        "session_id": crawl_session_id,
        "total_pdfs": 0,
        "vectorized": 0,
        "failed": 0,
        "skipped": 0,
        "duplicates": 0,
    }

    # Ensure database connection
    if not db_manager.is_connected:
        await db_manager.connect()

    websites_collection = db_manager.get_collection("websites")

    # Aggregation pipeline to extract unvectorized PDFs from nested structure
    pipeline = [
        {"$match": {"crawlSessionId": crawl_session_id}},
        {"$unwind": "$visitedUrls"},
        {"$unwind": "$visitedUrls.documents"},
        {"$match": {
            "visitedUrls.documents.documentType": "pdf",
            "visitedUrls.documents.isVectorized": "0",
            "visitedUrls.documents.isDeleted": {"$ne": True},
        }},
        {"$project": {
            "_id": 1,
            "websiteUrl": 1,
            "crawlSessionId": 1,
            "visitedUrl": "$visitedUrls.url",
            "doc": "$visitedUrls.documents",
        }},
    ]

    cursor = websites_collection.aggregate(pipeline)
    pdf_docs = await cursor.to_list(length=None)
    results["total_pdfs"] = len(pdf_docs)

    if not pdf_docs:
        logger.info(f"No unvectorized PDFs found for session {crawl_session_id}")
        return results

    logger.info(f"Found {len(pdf_docs)} unvectorized PDFs for session {crawl_session_id}")

    # Process each PDF through the Process_Docling pipeline
    async with semaphore:
        _documentProcessor = await AsyncDocumentProcessor.from_config()

        for pdf_info in pdf_docs:
            doc = pdf_info["doc"]
            file_id = doc.get("fileId", "")
            file_name = doc.get("originalFile", "unknown.pdf")
            file_path = doc.get("filePath", "")
            website_url = pdf_info.get("websiteUrl", "")
            visited_url = pdf_info.get("visitedUrl", "")

            if not file_path or not os.path.exists(file_path):
                # Try fallback path using UPLOAD_DIR + fileId
                fallback_path = os.path.join(app_config.UPLOAD_DIR, file_id)
                if os.path.exists(fallback_path):
                    file_path = fallback_path
                else:
                    logger.error(f"PDF file not found: {file_path} (nor {fallback_path}). Skipping {file_id}.")
                    await _update_pdf_vectorization_status(
                        websites_collection, website_url, crawl_session_id,
                        visited_url, file_id, status="-1"
                    )
                    results["failed"] += 1
                    continue

            logger.info(f"Processing PDF: {file_name} ({file_id})")

            try:
                process_result = await _documentProcessor.process_document_main(
                    file_id, file_name, file_path
                )

                if process_result is None or (isinstance(process_result, dict) and process_result.get("status") == "failed"):
                    logger.error(f"Processing failed for {file_id}")
                    await _update_pdf_vectorization_status(
                        websites_collection, website_url, crawl_session_id,
                        visited_url, file_id, status="-1"
                    )
                    results["failed"] += 1
                    continue

                if isinstance(process_result, dict):
                    status = process_result.get("status", "")

                    if status == "duplicate":
                        logger.info(f"File {file_id} already vectorized (duplicates)")
                        await _update_pdf_vectorization_status(
                            websites_collection, website_url, crawl_session_id,
                            visited_url, file_id, status="1"
                        )
                        results["duplicates"] += 1
                        continue

                    if status == "partial_failure":
                        logger.warning(f"Partial failure for {file_id}")
                        await _update_pdf_vectorization_status(
                            websites_collection, website_url, crawl_session_id,
                            visited_url, file_id, status="4"
                        )
                        results["failed"] += 1
                        continue

                    if status == "inserted":
                        logger.info(f"Successfully vectorized {file_id} ({process_result.get('count', 0)} chunks)")
                        await _update_pdf_vectorization_status(
                            websites_collection, website_url, crawl_session_id,
                            visited_url, file_id, status="1"
                        )
                        results["vectorized"] += 1
                        continue

                # Fallback for non-dict return (older processor returns file_id string on success)
                if process_result:
                    await _update_pdf_vectorization_status(
                        websites_collection, website_url, crawl_session_id,
                        visited_url, file_id, status="1"
                    )
                    results["vectorized"] += 1
                else:
                    await _update_pdf_vectorization_status(
                        websites_collection, website_url, crawl_session_id,
                        visited_url, file_id, status="-1"
                    )
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"Exception processing {file_id}: {e}", exc_info=True)
                await _update_pdf_vectorization_status(
                    websites_collection, website_url, crawl_session_id,
                    visited_url, file_id, status="-1"
                )
                results["failed"] += 1

    logger.info(
        f"Vectorization complete for session {crawl_session_id}: "
        f"{results['vectorized']} vectorized, {results['failed']} failed, "
        f"{results['duplicates']} duplicates out of {results['total_pdfs']} total"
    )
    return results


async def _update_pdf_vectorization_status(
    websites_collection,
    website_url: str,
    crawl_session_id: str,
    visited_url: str,
    file_id: str,
    status: str = "1",
):
    """
    Update the isVectorized field for a specific PDF in the nested websites collection.

    Uses MongoDB's arrayFilters to target the exact nested document.
    """
    try:
        update_result = await websites_collection.update_one(
            {
                "websiteUrl": website_url,
                "crawlSessionId": crawl_session_id,
            },
            {
                "$set": {
                    "visitedUrls.$[url].documents.$[doc].isVectorized": status,
                    "visitedUrls.$[url].documents.$[doc].updatedAt": datetime.now(timezone.utc),
                    "visitedUrls.$[url].documents.$[doc].status": (
                        "vectorized" if status == "1"
                        else "failed" if status == "-1"
                        else "partial_failure" if status == "4"
                        else "pending"
                    ),
                }
            },
            array_filters=[
                {"url.url": visited_url},
                {"doc.fileId": file_id},
            ],
        )

        if update_result.modified_count > 0:
            logger.debug(f"Updated isVectorized={status} for {file_id} in websites collection")
        else:
            logger.warning(f"No document matched for status update: {file_id} in {visited_url}")

    except Exception as e:
        logger.error(f"Failed to update vectorization status for {file_id}: {e}", exc_info=True)
