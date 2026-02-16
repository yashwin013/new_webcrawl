"""
Document Processing API Router.

Endpoints:
    POST   /api/v1/documents/upload          - Upload a PDF for processing
    POST   /api/v1/documents/process         - Process a document from storage
    POST   /api/v1/documents/batch-process   - Process multiple documents
    GET    /api/v1/documents/{document_id}   - Get document details
    GET    /api/v1/documents/{document_id}/status - Get processing status
    DELETE /api/v1/documents/{document_id}   - Soft delete a document
    GET    /api/v1/documents                 - List all documents (with filters)
"""

import asyncio
import uuid
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query

from app.api.models import (
    DocumentUploadResponse,
    DocumentProcessRequest,
    DocumentBatchProcessRequest,
    DocumentProcessResponse,
    DocumentBatchProcessResponse,
    DocumentDetailResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    TaskStatus,
    ErrorResponse,
)
from app.api.task_manager import TaskManager
from app.config import get_logger, UPLOAD_DIR, PDF_STORAGE_PATH
from app.schemas.document import DocumentStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


def _get_document_store():
    """Lazy-load DocumentStore to avoid import-time side effects."""
    from app.services.document_store import DocumentStore
    return DocumentStore.from_config()


def _crawled_doc_to_detail(doc) -> DocumentDetailResponse:
    """Convert a CrawledDocument to DocumentDetailResponse."""
    return DocumentDetailResponse(
        file_id=doc.file_id,
        original_file=doc.original_file,
        source_url=doc.source_url,
        file_path=doc.file_path,
        document_type=getattr(doc, "document_type", "pdf"),
        status=doc.status if isinstance(doc.status, str) else doc.status.value,
        vector_count=doc.vector_count,
        file_size=doc.file_size,
        page_count=doc.page_count,
        mime_type=doc.mime_type,
        crawl_session_id=doc.crawl_session_id,
        is_deleted=doc.is_deleted,
        is_crawled=doc.is_crawled,
        is_vectorized=doc.is_vectorized,
        is_ocr_required=doc.is_ocr_required,
        is_ocr_completed=doc.is_ocr_completed,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


# ======================== Background Task Runners ========================

async def _run_process_document(task_id: str, file_id: str, file_path: str, store_vectors: bool):
    """Background task: process a single document with Docling."""
    from app.docling import AsyncDocumentProcessor
    from app.core.lifecycle import startup_app

    tm = TaskManager.get_instance()
    tm.update_task(task_id, status=TaskStatus.RUNNING, started_at=datetime.utcnow())
    start_time = time.time()

    try:
        await startup_app()
        processor = await AsyncDocumentProcessor.from_config()

        result = await processor.process_document_main(
            file_id=file_id,
            file_name=Path(file_path).name,
            pdfpath=file_path,
        )

        elapsed = time.time() - start_time

        if result:
            chunks = result.get("chunks", 0)
            tm.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                total_chunks=chunks,
                documents_processed=1,
            )

            # Update document store status
            try:
                store = _get_document_store()
                await asyncio.to_thread(store.mark_vectorized, file_id, vector_count=chunks)
            except Exception:
                pass

            logger.info(f"Process task {task_id} completed: {chunks} chunks in {elapsed:.1f}s")
        else:
            tm.update_task(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=datetime.utcnow(),
                error="Processing returned no result",
                documents_failed=1,
            )

        try:
            await processor.close()
        except Exception:
            pass

    except Exception as e:
        tm.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.utcnow(),
            error=str(e),
            documents_failed=1,
        )
        logger.error(f"Process task {task_id} failed: {e}", exc_info=True)


async def _run_batch_process(
    task_id: str,
    file_ids: Optional[list[str]],
    folder_path: Optional[str],
    store_vectors: bool,
):
    """Background task: process multiple documents."""
    from app.docling import AsyncDocumentProcessor
    from app.core.lifecycle import startup_app

    tm = TaskManager.get_instance()
    tm.update_task(task_id, status=TaskStatus.RUNNING, started_at=datetime.utcnow())

    processed_count = 0
    failed_count = 0
    total_chunks = 0

    try:
        await startup_app()
        processor = await AsyncDocumentProcessor.from_config()

        # Collect files to process
        files_to_process: list[tuple[str, str]] = []  # (file_id, file_path)

        if folder_path:
            folder = Path(folder_path)
            if folder.exists():
                for pdf in folder.glob("*.pdf"):
                    files_to_process.append((pdf.stem, str(pdf)))

        if file_ids:
            store = _get_document_store()
            for fid in file_ids:
                doc = await asyncio.to_thread(store.get_by_file_id, fid)
                if doc and doc.file_path:
                    files_to_process.append((fid, doc.file_path))
                else:
                    # Check default upload dir
                    default_path = Path(UPLOAD_DIR) / fid
                    if default_path.exists():
                        files_to_process.append((fid, str(default_path)))

        tm.update_task(task_id, total_documents=len(files_to_process))

        for file_id, file_path in files_to_process:
            try:
                result = await processor.process_document_main(
                    file_id=file_id,
                    file_name=Path(file_path).name,
                    pdfpath=file_path,
                )
                if result:
                    chunks = result.get("chunks", 0)
                    total_chunks += chunks
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {file_id}: {e}")
                failed_count += 1

            tm.update_task(
                task_id,
                documents_processed=processed_count,
                documents_failed=failed_count,
                total_chunks=total_chunks,
            )

        tm.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.utcnow(),
        )

        try:
            await processor.close()
        except Exception:
            pass

        logger.info(
            f"Batch process {task_id} completed: {processed_count} processed, "
            f"{failed_count} failed, {total_chunks} chunks"
        )

    except Exception as e:
        tm.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.utcnow(),
            error=str(e),
        )
        logger.error(f"Batch process {task_id} failed: {e}", exc_info=True)


# ======================== Endpoints ========================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a PDF for processing",
    description="Upload a PDF file and register it in the document store.",
    responses={400: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    source_url: str = Form(default="upload", description="Source URL or identifier"),
    crawl_session_id: str = Form(default="", description="Crawl session ID"),
):
    """Upload a PDF file for processing."""
    # Validate file type
    if file.content_type and file.content_type != "application/pdf":
        if not (file.filename and file.filename.lower().endswith(".pdf")):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Generate file ID (used for both disk storage and document store)
    file_id = f"{uuid.uuid4()}.pdf"
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file_id

    # Save file
    try:
        content = await file.read()
        file_size = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Register in document store (use same file_id for disk and mongo)
    try:
        store = _get_document_store()
        doc = await asyncio.to_thread(
            store.create_document,
            original_file=file.filename or file_id,
            source_url=source_url,
            file_path=str(file_path),
            crawl_session_id=crawl_session_id or f"upload_{uuid.uuid4().hex[:8]}",
            file_size=file_size,
            file_id=file_id,
        )
        # Confirm the store used our file_id
        file_id = doc.file_id
    except Exception as e:
        logger.error(f"Failed to register document: {e}")
        # Still return success since file was saved
        pass

    return DocumentUploadResponse(
        document_id=file_id,
        file_name=file.filename or file_id,
        file_size=file_size,
        status="pending",
        message="Document uploaded successfully",
    )


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    summary="Process a document from storage",
    description="Start processing a document (chunking + vectorization) in the background.",
    responses={404: {"model": ErrorResponse}},
)
async def process_document(request: DocumentProcessRequest):
    """Process a document from storage."""
    # Resolve file path
    file_path = request.file_path
    if not file_path:
        # Try default locations on disk
        for base_dir in [Path(UPLOAD_DIR), PDF_STORAGE_PATH]:
            candidate = base_dir / request.file_id
            if candidate.exists():
                file_path = str(candidate)
                break

    # If still not found, look up the stored path from the document store
    if not file_path or not Path(file_path).exists():
        try:
            store = _get_document_store()
            doc = await asyncio.to_thread(store.get_by_file_id, request.file_id)
            if doc and doc.file_path and Path(doc.file_path).exists():
                file_path = doc.file_path
        except Exception:
            pass

    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found for file_id '{request.file_id}'. Provide file_path.",
        )

    task_id = str(uuid.uuid4())
    tm = TaskManager.get_instance()
    task = tm.create_task(task_id=task_id, task_type="process")

    async_task = asyncio.create_task(
        _run_process_document(task_id, request.file_id, file_path, request.store_vectors)
    )
    task._async_task = async_task

    return DocumentProcessResponse(
        task_id=task_id,
        file_id=request.file_id,
        status="pending",
        message=f"Processing started for {request.file_id}",
    )


@router.post(
    "/batch-process",
    response_model=DocumentBatchProcessResponse,
    summary="Process multiple documents",
    description="Start batch processing of multiple documents in the background.",
    responses={400: {"model": ErrorResponse}},
)
async def batch_process_documents(request: DocumentBatchProcessRequest):
    """Process multiple documents in batch."""
    if not request.file_ids and not request.folder_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file_ids' or 'folder_path'",
        )

    if request.folder_path and not Path(request.folder_path).exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    task_id = str(uuid.uuid4())
    tm = TaskManager.get_instance()
    task = tm.create_task(task_id=task_id, task_type="batch_process")

    async_task = asyncio.create_task(
        _run_batch_process(task_id, request.file_ids, request.folder_path, request.store_vectors)
    )
    task._async_task = async_task

    total = 0
    if request.file_ids:
        total += len(request.file_ids)
    if request.folder_path:
        total += len(list(Path(request.folder_path).glob("*.pdf")))

    return DocumentBatchProcessResponse(
        task_id=task_id,
        status="pending",
        message=f"Batch processing started for {total} documents",
        total_documents=total,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Get full details of a document by its file ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_document(document_id: str):
    """Get document details by file ID."""
    store = _get_document_store()
    doc = await asyncio.to_thread(store.get_by_file_id, document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return _crawled_doc_to_detail(doc)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
    description="Get processing status, vector count, and page count for a document.",
    responses={404: {"model": ErrorResponse}},
)
async def get_document_status(document_id: str):
    """Get processing status of a document."""
    store = _get_document_store()
    doc = await asyncio.to_thread(store.get_by_file_id, document_id)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return DocumentStatusResponse(
        file_id=doc.file_id,
        status=doc.status if isinstance(doc.status, str) else doc.status.value,
        vector_count=doc.vector_count,
        page_count=doc.page_count,
        is_vectorized=doc.is_vectorized,
        is_ocr_required=doc.is_ocr_required,
        is_ocr_completed=doc.is_ocr_completed,
        error_message=getattr(doc, "error_message", None),
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Soft delete a document",
    description="Marks a document as deleted (soft delete). Does not remove files.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(document_id: str):
    """Soft delete a document."""
    store = _get_document_store()

    # Check if document exists
    doc = await asyncio.to_thread(store.get_by_file_id, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    deleted = await asyncio.to_thread(store.soft_delete, document_id)

    return DeleteDocumentResponse(
        file_id=document_id,
        deleted=deleted,
        message="Document deleted successfully" if deleted else "Failed to delete document",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="List documents with optional filters for status, type, and session.",
)
async def list_documents(
    status: Optional[str] = Query(default=None, description="Filter by status (pending, processing, vectorized, failed)"),
    document_type: Optional[str] = Query(default=None, description="Filter by type (html, pdf)"),
    crawl_session_id: Optional[str] = Query(default=None, description="Filter by crawl session ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """List documents with optional filters."""
    store = _get_document_store()
    collection = await asyncio.to_thread(store._get_collection)

    # Build query filter
    query_filter: dict = {"isDeleted": False}

    if status:
        query_filter["status"] = status
    if document_type:
        query_filter["documentType"] = document_type
    if crawl_session_id:
        query_filter["crawlSessionId"] = crawl_session_id

    # Get total count
    total = await asyncio.to_thread(collection.count_documents, query_filter)

    # Fetch documents with pagination
    from pymongo import DESCENDING
    
    def _fetch_docs():
        cursor = (
            collection.find(query_filter)
            .sort("createdAt", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        return list(cursor)
    
    raw_docs = await asyncio.to_thread(_fetch_docs)

    from app.schemas.document import CrawledDocument
    documents = []
    for raw_doc in raw_docs:
        try:
            doc = CrawledDocument.from_mongo_dict(raw_doc)
            documents.append(_crawled_doc_to_detail(doc))
        except Exception as e:
            logger.warning(f"Skipping malformed document: {e}")
    
    return DocumentListResponse(
        documents=documents,
        total=total,
        limit=limit,
        offset=offset,
    )
