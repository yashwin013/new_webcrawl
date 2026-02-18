"""
Web Crawl RAG Pipeline - Main Application Entrypoint

This module provides a unified interface for:
1. Web crawling and PDF extraction
2. Document processing with Docling
3. Vector storage in Qdrant
4. RAG query interface

Usage:
    # As a module
    from app import settings
    from app.crawling import WebScrapingPipeline, run_pipeline
    from app.docling import AsyncDocumentProcessor
    
    # As a script
    python app.py --help
"""

import asyncio
from pathlib import Path
from typing import Optional
import atexit
import signal
import sys

# Import configuration and logging first
from app.config import (
    settings,
    app_config,
    get_logger,
    get_qdrant_client,
    get_embedding_model,
    ensure_output_dirs,
    close_connections,
)

# Import lifecycle management
from app.core.lifecycle import startup_app, shutdown_app, health_check

logger = get_logger(__name__)

# Track if lifecycle was initialized
_lifecycle_initialized = False


def run_crawler(
    start_url: str, 
    output_dir: Optional[Path] = None,
    max_pages: int = 50,
    max_depth: int = 3,
) -> None:
    """
    Run the web crawler pipeline on a URL.
    
    Args:
        start_url: URL to start crawling from
        output_dir: Directory to save outputs
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
    """
    from app.crawling import run_pipeline, PipelineConfig, CrawlConfig
    
    config = PipelineConfig(
        output_dir=output_dir or Path("outputs/scraped"),
        crawl=CrawlConfig(max_pages=max_pages, max_depth=max_depth),
    )
    
    async def _run():
        return await run_pipeline(start_url, config=config)
    
    result = asyncio.run(_run())
    logger.info(f"Crawled {result.total_pages} pages, {result.total_chunks} chunks")
    return result


async def process_documents(
    pdf_folder: Path,
    store: bool = False,
) -> dict:
    """
    Process PDFs with Docling and optionally store to Qdrant.
    
    Args:
        pdf_folder: Folder containing PDF files
        store: Whether to store vectors to Qdrant
        
    Returns:
        Processing results summary
    """
    from app.docling import AsyncDocumentProcessor
    
    # Initialize lifecycle if not already done
    global _lifecycle_initialized
    if not _lifecycle_initialized:
        await startup_app()
        _lifecycle_initialized = True
    
    ensure_output_dirs()
    
    processor = await AsyncDocumentProcessor.from_config()
    
    results = {
        "processed": [],
        "failed": [],
        "total_chunks": 0,
    }
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    try:
        for pdf_path in pdf_files:
            try:
                file_id = pdf_path.stem
                result = await processor.process_document_main(
                    file_id=file_id,
                    file_name=pdf_path.name,
                    pdfpath=str(pdf_path),
                )
                if result:
                    results["processed"].append(pdf_path.name)
                    results["total_chunks"] += result.get("chunks", 0)
                else:
                    results["failed"].append(pdf_path.name)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results["failed"].append(pdf_path.name)
    finally:
        # Clean up processor resources after all PDFs are processed
        try:
            await processor.close()
        except Exception as e:
            logger.error(f"Error closing processor: {e}")
    
    return results


async def query_rag(
    query: str, 
    top_k: int = 5,
    file_id: Optional[str] = None,
    generate_answer: bool = True,
) -> dict:
    """
    Query the RAG system and optionally generate an answer using LLM.
    
    Args:
        query: Search query
        top_k: Number of chunks to retrieve for context
        file_id: Optional file filter
        generate_answer: Whether to generate an LLM answer (default: True)
        
    Returns:
        Dict with 'answer' (LLM response) and 'sources' (retrieved chunks)
    """
    from app.docling import DoclingQdrantService
    import openai
    import os
    
    # Initialize lifecycle if not already done
    global _lifecycle_initialized
    if not _lifecycle_initialized:
        await startup_app()
        _lifecycle_initialized = True
    
    # Retrieve relevant chunks
    service = await DoclingQdrantService.from_config()

    print("hello")
    chunks = await service.search_vectors_async(
        query=query,
        top_k=top_k,
        file_id=file_id,
    )
    
    if not generate_answer:
        return {"answer": None, "sources": chunks}
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        score = chunk.get('score', 0)
        context_parts.append(f"[Source {i}] (relevance: {score:.2f})\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for LLM
    system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Your answers should be:
- Accurate and based only on the provided context
- Well-structured and easy to read
- Comprehensive but concise
- If the context doesn't contain enough information, say so clearly

Do not make up information. Only use what is provided in the context."""

    user_prompt = f"""Context from documents:
{context}

Question: {query}

Please provide a detailed and accurate answer based on the above context."""

    # Call OpenAI API
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = f"Error generating answer: {e}"
    
    return {"answer": answer, "sources": chunks}


def main():
    """Main entry point when running as script."""
    import argparse
    
    # Setup cleanup handlers
    def cleanup_handler(signum=None, frame=None):
        """Handle cleanup on exit."""
        global _lifecycle_initialized
        if _lifecycle_initialized:
            logger.info("Shutting down application...")
            try:
                asyncio.run(shutdown_app())
                _lifecycle_initialized = False
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        if signum is not None:
            sys.exit(0)
    
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    parser = argparse.ArgumentParser(
        description="Web Crawl RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py crawl https://example.com
    python app.py process ./pdfs --store
    python app.py query "What is machine learning?"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website")
    crawl_parser.add_argument("url", help="URL to start crawling")
    crawl_parser.add_argument("-o", "--output", help="Output directory")
    crawl_parser.add_argument("--max-pages", type=int, default=50)
    crawl_parser.add_argument("--max-depth", type=int, default=3)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDF documents")
    process_parser.add_argument("folder", help="Folder containing PDFs")
    process_parser.add_argument("--store", action="store_true", help="Store to Qdrant")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("-f", "--file-id", help="Filter by file ID")
    query_parser.add_argument("--raw", action="store_true", help="Show raw chunks instead of LLM answer")
    query_parser.add_argument("--sources", action="store_true", help="Show source chunks with answer")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process PDFs into RAG vector database")
    batch_parser.add_argument("pdf_folder", help="Folder containing PDF files to process")
    batch_parser.add_argument("-o", "--output", help="Output directory (default: outputs/batch/)")
    batch_parser.add_argument("--skip-existing", action="store_true", help="Skip already processed PDFs")
    batch_parser.add_argument("--max", type=int, help="Maximum number of PDFs to process")
    batch_parser.add_argument("--store", action="store_true", help="Store chunks to Qdrant after processing")
    batch_parser.add_argument("--parallel", action="store_true", help="Use parallel processing (faster, more memory)")
    batch_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    batch_parser.add_argument("--docling", action="store_true", help="Use Docling for high-quality PDF processing")
    
    # Orchestrator command (NEW - Multi-site parallel crawling)
    orchestrator_parser = subparsers.add_parser("orchestrator", help="Run multi-site parallel crawler")
    orchestrator_parser.add_argument("urls", nargs="+", help="URLs to crawl (space-separated)")
    orchestrator_parser.add_argument("--max-pages", type=int, default=50, help="Max pages per site")
    orchestrator_parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth")
    orchestrator_parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    
    args = parser.parse_args()
    
    if args.command == "crawl":
        output_dir = Path(args.output) if args.output else None
        run_crawler(args.url, output_dir, args.max_pages, args.max_depth)
        
    elif args.command == "process":
        async def run_process():
            results = await process_documents(Path(args.folder), args.store)
            return results
        
        results = asyncio.run(run_process())
        print(f"\nProcessed: {len(results['processed'])} files")
        print(f"Failed: {len(results['failed'])} files")
        print(f"Total chunks: {results['total_chunks']}")
        
    elif args.command == "query":
        async def run_query():
            generate = not args.raw
            result = await query_rag(args.query, args.top_k, args.file_id, generate)
            return result
        
        result = asyncio.run(run_query())
        
        if args.raw:
            # Show raw chunks
            for i, chunk in enumerate(result["sources"], 1):
                print(f"\n--- Result {i} ---")
                print(f"Score: {chunk.get('score', 'N/A')}")
                text = chunk.get('text', '')[:200]
                # Handle Unicode encoding for Windows console
                safe_text = text.encode('ascii', errors='replace').decode('ascii')
                print(f"Text: {safe_text}...")
        else:
            # Show LLM answer
            print(f"\n{'='*60}")
            print("ANSWER:")
            print('='*60)
            print(result["answer"])
            
            if args.sources:
                print(f"\n{'='*60}")
                print(f"SOURCES ({len(result['sources'])} chunks used):")
                print('='*60)
                for i, chunk in enumerate(result["sources"], 1):
                    print(f"\n[{i}] Score: {chunk.get('score', 0):.3f}")
                    text = chunk.get('text', '')[:150]
                    # Handle Unicode encoding for Windows console
                    safe_text = text.encode('ascii', errors='replace').decode('ascii')
                    print(f"    {safe_text}...")
    
    elif args.command == "batch":
        from app.services.batch_processor import (
            process_batch,
            store_batch_to_qdrant,
            BatchConfig,
            BATCH_CHUNK_MIN_TOKENS,
            BATCH_CHUNK_MAX_TOKENS,
        )
        
        pdf_folder = Path(args.pdf_folder)
        if not pdf_folder.exists():
            print(f"Error: Folder not found: {pdf_folder}")
            return
        
        output_dir = Path(args.output) if args.output else None
        
        config = BatchConfig(
            min_tokens=BATCH_CHUNK_MIN_TOKENS,
            max_tokens=BATCH_CHUNK_MAX_TOKENS,
            max_workers=args.workers,
            use_docling=args.docling,
        )
        
        print(f"\n{'='*60}")
        print("  BATCH PDF PROCESSING")
        print(f"{'='*60}")
        print(f"  Source: {pdf_folder}")
        print(f"  Skip existing: {args.skip_existing}")
        print(f"  Max PDFs: {args.max or 'All'}")
        print(f"  Parallel: {args.parallel} (workers: {args.workers})")
        print(f"  Docling: {args.docling}")
        print(f"  Chunk settings: min={config.min_tokens}, max={config.max_tokens}")
        print(f"{'='*60}\n")
        
        result = process_batch(
            pdf_folder,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
            max_pdfs=args.max,
            config=config,
            parallel=args.parallel,
        )
        
        print(f"\n{'='*60}")
        print("  PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  PDFs processed: {len(result.pdfs_processed)}")
        print(f"  PDFs failed: {len(result.pdfs_failed)}")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"    - Parents: {result.total_parents}")
        print(f"    - Children: {result.total_children}")
        
        if result.pdfs_failed:
            print(f"\n  Failed PDFs:")
            for pdf in result.pdfs_failed[:5]:
                print(f"    - {pdf}")
            if len(result.pdfs_failed) > 5:
                print(f"    ... and {len(result.pdfs_failed) - 5} more")
        
        if args.store and result.total_chunks > 0:
            print(f"\n  Storing to Qdrant...")
            chunks_file = (output_dir or Path("outputs/batch")) / "all_chunks.json"
            store_batch_to_qdrant(chunks_file)
            print(f"  ✓ Stored to Qdrant")
        
        print()
    
    elif args.command == "orchestrator":
        from app.orchestrator.coordinator import MultiSiteOrchestrator
        from app.orchestrator.config import get_default_config
        from scripts.process_ocr_backlog import process_ocr_backlog
        from pathlib import Path
        
        async def run_orchestrator():
            config = get_default_config()
            config.enable_monitoring = not args.no_monitoring
            
            orchestrator = MultiSiteOrchestrator(config)
            
            try:
                print(f"\n{'='*70}")
                print(f"MULTI-SITE PARALLEL CRAWLER")
                print(f"{'='*70}")
                print(f"Websites: {len(args.urls)}")
                for url in args.urls:
                    print(f"  - {url}")
                print(f"{'='*70}\n")
                
                await orchestrator.startup()
                
                # Generate session ID here so we can use it for post-crawl vectorization
                import uuid as _uuid
                from datetime import datetime as _dt
                crawl_session_id = f"session-{_dt.utcnow().strftime('%Y%m%d-%H%M%S')}-{_uuid.uuid4().hex[:8]}"
                print(f"Crawl Session ID: {crawl_session_id}")
                
                await orchestrator.crawl_websites(
                    args.urls,
                    max_pages_per_site=args.max_pages,
                    max_depth=args.max_depth,
                    crawl_session_id=crawl_session_id,
                )
                
                # Results
                stats = orchestrator.stats
                progress = orchestrator.get_progress()
                
                print(f"\n{'='*70}")
                print("CRAWLING RESULTS")
                print(f"{'='*70}")
                print(f"Duration: {stats.duration_seconds:.1f}s")
                print(f"Websites completed: {stats.websites_completed}")
                print(f"Websites failed: {stats.websites_failed}")
                
                if progress:
                    print(f"Tasks processed: {progress['tasks']['processed']}")
                    print(f"Tasks failed: {progress['tasks']['failed']}")
                    if progress['tasks']['processed'] > 0:
                        print(f"Success rate: {progress['tasks']['success_rate']:.1f}%")
                
                print(f"{'='*70}\n")
                
                # === Post-Crawl: Vectorize PDFs via Process_Docling pipeline ===
                # Runs BEFORE shutdown() so DB connections are still alive
                try:
                    from app.docling.pipeline import vectorize_crawled_pdfs
                    print(f"\n{'='*70}")
                    print("POST-CRAWL PDF VECTORIZATION (Process_Docling)")
                    print(f"Session: {crawl_session_id}")
                    print(f"{'='*70}\n")
                    vresults = await vectorize_crawled_pdfs(crawl_session_id)
                    print(f"\n{'='*70}")
                    print("VECTORIZATION SUMMARY")
                    print(f"{'='*70}")
                    print(f"  Total PDFs:      {vresults.get('total_pdfs', 0)}")
                    print(f"  Vectorized:      {vresults.get('vectorized', 0)}")
                    print(f"  Duplicates:      {vresults.get('duplicates', 0)}")
                    print(f"  Failed:          {vresults.get('failed', 0)}")
                    print(f"{'='*70}\n")
                except Exception as e:
                    print(f"\n⚠ Vectorization failed: {e}")
                    import traceback; traceback.print_exc()
                    print("Crawl data is saved. Retry vectorization manually if needed.")
                
            finally:
                await orchestrator.shutdown()
            
            # Phase 2: Process OCR backlog
            backlog_file = Path("outputs/ocr_backlog/pending_ocr.jsonl")
            if backlog_file.exists():
                print(f"\n{'='*70}")
                print("PHASE 2: OCR BACKLOG PROCESSING")
                print(f"{'='*70}\n")
                
                try:
                    await process_ocr_backlog()
                except Exception as e:
                    print(f"\n❌ OCR processing failed: {e}")
                    print("You can retry later with: python scripts/process_ocr_backlog.py")
            else:
                print("\n✓ No pages require OCR processing")
        
        asyncio.run(run_orchestrator())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
