"""
FastAPI Main Application.

Registers all API routers and provides the application entry point.

Usage:
    uvicorn app.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.crawl_router import router as crawl_router
from app.api.document_router import router as document_router
from app.api.models import HealthResponse
from app.config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting FastAPI application...")
    logger.info("=" * 60)

    try:
        from app.core.lifecycle import startup_app
        await startup_app()
        logger.info("Application lifecycle initialized")
    except Exception as e:
        logger.warning(f"Lifecycle startup warning (non-fatal): {e}")

    from app.config import ensure_output_dirs
    ensure_output_dirs()

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    try:
        from app.core.lifecycle import shutdown_app
        await shutdown_app()
    except Exception as e:
        logger.warning(f"Shutdown warning: {e}")
    logger.info("Application shut down")


# ======================== Create FastAPI App ========================

app = FastAPI(
    title="Web Crawl RAG Pipeline API",
    description=(
        "API for web crawling, document processing, and RAG pipeline.\n\n"
        "## Features\n"
        "- **Crawling**: Start single or batch crawls with progress tracking\n"
        "- **Documents**: Upload, process, and manage PDF documents\n"
        "- **Background Tasks**: All heavy operations run asynchronously\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ======================== CORS Middleware ========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================== Include Routers ========================

app.include_router(crawl_router)
app.include_router(document_router)


# ======================== Root & Health Endpoints ========================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API information."""
    return {
        "name": "Web Crawl RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "crawl": {
                "POST /api/v1/crawl/start": "Start crawling a single URL",
                "POST /api/v1/crawl/batch": "Start crawling multiple websites",
                "GET /api/v1/crawl/status/{task_id}": "Get crawl task status",
                "POST /api/v1/crawl/stop/{task_id}": "Stop an active crawl",
                "GET /api/v1/crawl/results/{task_id}": "Get crawl results",
            },
            "documents": {
                "POST /api/v1/documents/upload": "Upload a PDF for processing",
                "POST /api/v1/documents/process": "Process a document from storage",
                "POST /api/v1/documents/batch-process": "Process multiple documents",
                "GET /api/v1/documents/{document_id}": "Get document details",
                "GET /api/v1/documents/{document_id}/status": "Get processing status",
                "DELETE /api/v1/documents/{document_id}": "Soft delete a document",
                "GET /api/v1/documents": "List all documents (with filters)",
            },
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@app.get("/api/v1/routes", tags=["Root"])
async def list_routes():
    """List all registered API routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": sorted(route.methods),
                "name": route.name,
                "tags": getattr(route, "tags", []),
            })
    return {"total_routes": len(routes), "routes": routes}


# ======================== Entry Point ========================

if __name__ == "__main__":
    import sys
    import asyncio

    # Windows: use ProactorEventLoop which supports subprocess creation.
    # Playwright launches a Node.js child process via asyncio.create_subprocess_exec,
    # which requires ProactorEventLoop on Windows.  Uvicorn's default setup forces
    # SelectorEventLoop (no subprocess support), so we override the policy here
    # and pass loop="none" to prevent uvicorn from resetting it.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        loop="none",  # preserve our ProactorEventLoop policy
    )
