"""
Web Crawling API Router.

Endpoints:
    POST /api/v1/crawl/start          - Start crawling a single URL
    POST /api/v1/crawl/batch          - Start crawling multiple websites
    GET  /api/v1/crawl/status/{id}    - Get crawl task status
    POST /api/v1/crawl/stop/{id}      - Stop an active crawl
    GET  /api/v1/crawl/results/{id}   - Get crawl results
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.api.models import (
    CrawlStartRequest,
    CrawlBatchRequest,
    CrawlTaskResponse,
    CrawlStatusResponse,
    CrawlResultsResponse,
    StopCrawlResponse,
    TaskStatus,
    ErrorResponse,
)
from app.api.task_manager import TaskManager, TaskInfo
from app.config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/crawl", tags=["Crawling"])


# ======================== Background Task Runners ========================

async def _run_single_crawl(task_id: str, url: str, max_pages: int, max_depth: int):
    """Background task: run the crawl pipeline for a single URL."""
    from app.crawling import run_pipeline, PipelineConfig, CrawlConfig

    tm = TaskManager.get_instance()
    tm.update_task(task_id, status=TaskStatus.RUNNING, started_at=datetime.utcnow())

    output_dir = Path(f"outputs/crawl_{task_id}")

    try:
        config = PipelineConfig(
            output_dir=output_dir,
            crawl=CrawlConfig(max_pages=max_pages, max_depth=max_depth),
        )
        result = await run_pipeline(url, config=config)

        tm.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            pages_crawled=result.pages_scraped,
            total_pages=result.total_pages,
            pages_failed=result.pages_failed,
            pdfs_downloaded=result.pdfs_downloaded,
            total_chunks=result.total_chunks,
            urls_visited=[p.url for p in result.pages],
            output_dir=str(output_dir),
            result=result,
        )
        logger.info(f"Crawl task {task_id} completed: {result.total_pages} pages, {result.total_chunks} chunks")

    except asyncio.CancelledError:
        tm.update_task(task_id, status=TaskStatus.CANCELLED, completed_at=datetime.utcnow())
        logger.info(f"Crawl task {task_id} was cancelled")
    except Exception as e:
        tm.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.utcnow(),
            error=str(e),
        )
        logger.error(f"Crawl task {task_id} failed: {e}", exc_info=True)


async def _run_batch_crawl(task_id: str, urls: list[str], max_pages: int, max_depth: int):
    """Background task: run the orchestrator for multiple URLs."""
    from app.orchestrator.coordinator import MultiSiteOrchestrator
    from app.orchestrator.config import get_default_config

    tm = TaskManager.get_instance()
    task = tm.get_task(task_id)
    
    tm.update_task(
        task_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.utcnow(),
        total_documents=len(urls),
    )

    orchestrator = None
    try:
        config = get_default_config()
        orchestrator = MultiSiteOrchestrator(config)
        
        # Store orchestrator reference for cancellation
        if task:
            task._orchestrator = orchestrator

        await orchestrator.startup()
        await orchestrator.crawl_websites(urls, max_pages_per_site=max_pages, max_depth=max_depth)

        stats = orchestrator.stats
        progress = orchestrator.get_progress()

        tm.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            pages_crawled=stats.total_tasks_processed,
            pages_failed=stats.total_tasks_failed,
            total_pages=stats.total_tasks_processed + stats.total_tasks_failed,
            urls_visited=urls,
            documents_processed=stats.websites_completed,
            documents_failed=stats.websites_failed,
        )

        await orchestrator.shutdown()
        logger.info(f"Batch crawl {task_id} completed: {stats.websites_completed} sites")

    except asyncio.CancelledError:
        # Gracefully shutdown orchestrator on cancellation
        if orchestrator:
            logger.info(f"Shutting down orchestrator for cancelled task {task_id}...")
            try:
                await asyncio.wait_for(orchestrator.shutdown(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Orchestrator shutdown timed out for {task_id}")
        
        tm.update_task(task_id, status=TaskStatus.CANCELLED, completed_at=datetime.utcnow())
        logger.info(f"Batch crawl {task_id} was cancelled")
        raise
    except Exception as e:
        tm.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.utcnow(),
            error=str(e),
        )
        logger.error(f"Batch crawl {task_id} failed: {e}", exc_info=True)


# ======================== Endpoints ========================

@router.post(
    "/start",
    response_model=CrawlTaskResponse,
    summary="Start crawling a single URL",
    description="Starts an async crawl task for the given URL. Returns a task ID for tracking progress.",
    responses={400: {"model": ErrorResponse}},
)
async def start_crawl(request: CrawlStartRequest):
    """Start crawling a single URL in the background."""
    task_id = request.crawl_session_id or str(uuid.uuid4())

    tm = TaskManager.get_instance()

    # Check if task_id already exists
    if tm.get_task(task_id):
        raise HTTPException(status_code=400, detail=f"Task with ID '{task_id}' already exists")

    # Create task record
    task = tm.create_task(task_id=task_id, task_type="crawl", url=request.url)

    # Launch background crawl
    async_task = asyncio.create_task(
        _run_single_crawl(task_id, request.url, request.max_pages, request.max_depth)
    )
    task._async_task = async_task

    return CrawlTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Crawl task created for {request.url}",
        url=request.url,
    )


@router.post(
    "/batch",
    response_model=CrawlTaskResponse,
    summary="Start crawling multiple websites",
    description="Starts an async batch crawl for multiple URLs using the orchestrator.",
    responses={400: {"model": ErrorResponse}},
)
async def start_batch_crawl(request: CrawlBatchRequest):
    """Start crawling multiple websites in the background."""
    task_id = str(uuid.uuid4())

    tm = TaskManager.get_instance()
    task = tm.create_task(task_id=task_id, task_type="batch_crawl", urls=request.urls)

    async_task = asyncio.create_task(
        _run_batch_crawl(task_id, request.urls, request.max_pages, request.max_depth)
    )
    task._async_task = async_task

    return CrawlTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Batch crawl created for {len(request.urls)} URLs",
        urls=request.urls,
    )


@router.get(
    "/status/{task_id}",
    response_model=CrawlStatusResponse,
    summary="Get crawl task status",
    description="Returns current status and progress of a crawl task.",
    responses={404: {"model": ErrorResponse}},
)
async def get_crawl_status(task_id: str):
    """Get the status of a crawl task."""
    tm = TaskManager.get_instance()
    task = tm.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return CrawlStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress={
            "pages_crawled": task.pages_crawled,
            "pdfs_downloaded": task.pdfs_downloaded,
            "total_chunks": task.total_chunks,
        },
        pages_crawled=task.pages_crawled,
        total_pages=task.total_pages,
        pages_failed=task.pages_failed,
        current_url=task.current_url,
        started_at=task.started_at,
        completed_at=task.completed_at,
        duration_seconds=task.duration_seconds,
        error=task.error,
    )


@router.post(
    "/stop/{task_id}",
    response_model=StopCrawlResponse,
    summary="Stop an active crawl",
    description="Cancels a running crawl task.",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def stop_crawl(task_id: str):
    """Stop an active crawl task."""
    tm = TaskManager.get_instance()
    task = tm.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    # If task is already in a terminal state, return appropriate message
    if task.status == TaskStatus.COMPLETED:
        return StopCrawlResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            message=f"Task already completed (cannot cancel)",
        )
    
    if task.status == TaskStatus.FAILED:
        return StopCrawlResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message=f"Task already failed (cannot cancel)",
        )
    
    if task.status == TaskStatus.CANCELLED:
        return StopCrawlResponse(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            message=f"Task already cancelled",
        )

    # Task is PENDING or RUNNING - try to cancel it
    cancelled = tm.cancel_task(task_id)

    return StopCrawlResponse(
        task_id=task_id,
        status=TaskStatus.CANCELLED if cancelled else task.status,
        message="Crawl task cancelled successfully" if cancelled else "Failed to cancel task",
    )


@router.get(
    "/results/{task_id}",
    response_model=CrawlResultsResponse,
    summary="Get crawl results",
    description="Returns final results of a completed crawl task.",
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def get_crawl_results(task_id: str):
    """Get the results of a completed crawl task."""
    tm = TaskManager.get_instance()
    task = tm.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    if task.status == TaskStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_id}' is still running. Check /status/{task_id} for progress.",
        )

    return CrawlResultsResponse(
        task_id=task.task_id,
        status=task.status,
        urls_visited=task.urls_visited,
        documents_extracted=task.pdfs_downloaded,
        total_chunks=task.total_chunks,
        pdfs_downloaded=task.pdfs_downloaded,
        pages_scraped=task.pages_crawled,
        pages_failed=task.pages_failed,
        duration_seconds=task.duration_seconds,
        output_dir=task.output_dir,
    )
