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
