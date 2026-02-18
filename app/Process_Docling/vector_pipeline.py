import asyncio
import os

import logging
from app.admin.knowledgebase.store import get_file_for_vector, update_vector_file,update_file_process_status
from app.bot.kbvectordb.doc_processor import DocumentProcessor
from app.bot.kbvectordb.enhanced_document_processor import EnhancedDocumentProcessor
# from app.bot.kbvectordb.enhanced_qdrant_service import EnhancedQdrantService
from app.bot.processdocling.docling_qdrant_service import DoclingQdrantService

# from app.bot.kbvectordb.qdrantdb import QdrantService
from app.bot.processdocling.docprocessor import AsyncDocumentProcessor
from app.config import app_config

logger = logging.getLogger("vector_processor")


async def create_vector_pipeline(createdby: str):
    """
    Initiate vector db pipeline creating
    """
    _documentProcessor = await AsyncDocumentProcessor.from_config()

    try:
        files = await get_file_for_vector(createdby)
    except Exception:
        logger.info("No files to process")
        print("No file to process")
        return

    if not files:
        print("No file to process")
        return

    for file_doc in files:
        file_path = os.path.join(app_config.UPLOAD_DIR, file_doc.fileId)
        file_name = file_doc.originalfile
        file_id = file_doc.fileId
        file_obj_id = file_doc.id

        # Mark status as "2" (processing started)
        await update_file_process_status(file_obj_id, file_id, createdby, "2")

        try:
            processfile = await _documentProcessor.process_document_main(
                file_id, file_name, file_path
            )

            # Case 1: Failed before touching Qdrant — safe to retry
            if processfile is None or processfile["status"] == "failed":
                logger.error(f"Processing failed for {file_path} (before Qdrant). Resetting to 1.")
                await update_file_process_status(file_obj_id, file_id, createdby, "1")
                continue

            # Case 2: Partial failure — some vectors may be in Qdrant already
            if processfile["status"] == "partial_failure":
                logger.error(f"Partial failure for {file_id}. Vectors may exist in Qdrant. Setting to 4.")
                # Don't reset to "1" — use "4" (needs manual attention or smart retry)
                await update_file_process_status(file_obj_id, file_id, createdby, "4")
                continue

            # Case 3: All duplicates — file already vectorized, mark as completed
            if processfile["status"] == "duplicate":
                logger.info(f"File {file_id} already vectorized (duplicates found). Marking as completed.")
                await update_vector_file(file_obj_id, file_id, createdby)
                print(f"Vector already exists for {file_id}, marked as completed.")
                continue

            # Case 4: New vectors inserted successfully — now update DB
            if processfile["status"] == "inserted":
                try:
                    count = await update_vector_file(file_obj_id, file_id, createdby)
                    if count <= 0:
                        logger.error(f"DB not updated for {file_id}. Setting status to 4 (needs attention).")
                        # DON'T reset to "1" — vectors are already in Qdrant!
                        # Use "4" = "vectors inserted but DB update failed"
                        await update_file_process_status(file_obj_id, file_id, createdby, "4")
                        continue
                    print(f"Vector inserted successfully for {file_id}")
                except Exception as db_err:
                    logger.error(f"DB update exception for {file_id}: {str(db_err)}", exc_info=True)
                    # Same: don't reset to "1", use "4"
                    await update_file_process_status(file_obj_id, file_id, createdby, "4")
                    continue

        except Exception as e:
            logger.error(f"Exception processing file {file_id}: {str(e)}", exc_info=True)
            await update_file_process_status(file_obj_id, file_id, createdby, "1")
            continue


async def create_vector_pipeline_old_11_02_2026_till_running(createdby:str):
    """
    Initiate vector db pipeline creating
    :return:
    """
    # _documentProcessor=await DocumentProcessor.from_config()
    # _documentProcessor=await EnhancedDocumentProcessor.from_config()

    _documentProcessor=await AsyncDocumentProcessor.from_config()

    files=await get_file_for_vector(createdby)
    if files:
       
        for file_doc in files:
            file_path = os.path.join(app_config.UPLOAD_DIR, file_doc.fileId) # app_config.UPLOAD_DIR + file_doc.fileId
            file_name=file_doc.originalfile
            file_id= file_doc.fileId
            file_obj_id=file_doc.id

            # processfile=_documentProcessor.process_and_store_pdf(file_id,file_name, file_path)
            # processfile=await asyncio.to_thread(
            # _documentProcessor.process_and_store_pdf,
            # file_id,
            # file_name,
            # file_path
            # )

            processfile=await _documentProcessor.process_document_main(file_id,
            file_name,
            file_path)
            if not processfile:
                logger.error(f" No vectors inserted for {file_path}. Skipping file.", exc_info=True)
                continue

            count=await update_vector_file(file_obj_id,file_id,createdby)    
            if count<=0:
                logger.error(f" db not update of  {file_id}. Skipping file.", exc_info=True)
                continue
            print(f"vector inserted successfully of {file_id}")
            # Vector creation completed successfully
    else:
       print(f"No file to process")
               

async def delete_file_vector_pipeline(fileid:str):
    """
    Initiate vector db pipeline delete the vector
    :return:
    """
    _qdrantService=await DoclingQdrantService.from_config()
    # _qdrantService=await QdrantService.from_config()
    result=_qdrantService.delete_by_file_id(fileid)
    if result>0:     
     print(f"permanent delete the vector of {fileid}")

    else:     
    logger.error(f"error when delete the vector of file {fileid} from vector db. Skipping file.", exc_info=True)
