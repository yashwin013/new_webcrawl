"""
MongoDB Document Store Service.

Provides CRUD operations for crawled PDF documents.
"""

from datetime import datetime
from typing import List, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import get_logger
from app.schemas.document import CrawledDocument, DocumentStatus, WebsiteCrawl, VisitedUrl, PdfDocument, PageDocument

logger = get_logger(__name__)


class DocumentStore:
    """
    MongoDB document store for crawled PDFs.
    
    Manages document metadata, processing status, and queries.
    Uses connection pooling via singleton pattern.
    
    Usage:
        store = DocumentStore.from_config()
        doc = store.create_document(...)
        pending = store.get_pending_documents()
    """
    
    _instance: Optional["DocumentStore"] = None
    _client: Optional[MongoClient] = None
    
    def __init__(self, mongodb_url: str, database_name: str):
        """
        Initialize document store.
        
        Args:
            mongodb_url: MongoDB connection URL
            database_name: Database name to use
        """
        self._mongodb_url = mongodb_url
        self._database_name = database_name
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
    
    def _get_client(self) -> MongoClient:
        """Get or create MongoDB client with connection pooling."""
        if DocumentStore._client is None:
            DocumentStore._client = MongoClient(
                self._mongodb_url,
                maxPoolSize=10,
                minPoolSize=1,
                serverSelectionTimeoutMS=5000,
            )
            logger.info(f"Connected to MongoDB: {self._database_name}")
        return DocumentStore._client
    
    def _get_collection(self) -> Collection:
        """Get the documents collection."""
        if self._collection is None:
            client = self._get_client()
            self._db = client[self._database_name]
            self._collection = self._db["documents"]
            self._ensure_indexes()
        return self._collection
    
    def _get_websites_collection(self) -> Collection:
        """Get the websites collection for nested structure."""
        client = self._get_client()
        if self._db is None:
            self._db = client[self._database_name]
        websites_collection = self._db["websites"]
        # Ensure indexes for websites collection
        websites_collection.create_index([("websiteUrl", ASCENDING), ("crawlSessionId", ASCENDING)], unique=True)
        websites_collection.create_index([("crawlSessionId", ASCENDING)])
        return websites_collection
    
    def _ensure_indexes(self) -> None:
        """Create indexes for efficient queries."""
        collection = self._collection
        if collection is not None:
            collection.create_index([("status", ASCENDING), ("createdAt", ASCENDING)])
            collection.create_index([("crawlSessionId", ASCENDING)])
            collection.create_index([("fileId", ASCENDING)], unique=True)
            collection.create_index([("sourceUrl", ASCENDING)])
            logger.info("MongoDB indexes ensured")
    
    @classmethod
    def from_config(cls) -> "DocumentStore":
        """Create DocumentStore from app configuration."""
        if cls._instance is None:
            from app.config import MONGODB_URL, MONGODB_DATABASE
            cls._instance = cls(MONGODB_URL, MONGODB_DATABASE)
        return cls._instance
    
    def create_document(
        self,
        original_file: str,
        source_url: str,
        file_path: str,
        crawl_session_id: str,
        file_size: int = 0,
        crawl_depth: int = 0,
        status: DocumentStatus = DocumentStatus.PENDING,
        file_id: Optional[str] = None,
    ) -> CrawledDocument:
        """
        Create a new document record.
        
        Args:
            original_file: Original filename
            source_url: URL where PDF was found
            file_path: Local storage path
            crawl_session_id: Session ID for this crawl
            file_size: File size in bytes
            crawl_depth: Depth in crawl tree
            status: Initial status (PENDING for chunking, STORED for storage only)
            file_id: Optional custom file ID (auto-generated if not provided)
            
        Returns:
            Created document
        """
        kwargs = dict(
            original_file=original_file,
            source_url=source_url,
            file_path=file_path,
            crawl_session_id=crawl_session_id,
            file_size=file_size,
            crawl_depth=crawl_depth,
            status=status,
        )
        if file_id:
            kwargs["file_id"] = file_id
        doc = CrawledDocument(**kwargs)
        
        collection = self._get_collection()
        mongo_doc = doc.to_mongo_dict()
        collection.insert_one(mongo_doc)
        
        logger.info(f"Created document: {doc.file_id} (status={status.value})")
        return doc
    
    def get_by_file_id(self, file_id: str) -> Optional[CrawledDocument]:
        """Get document by file ID."""
        collection = self._get_collection()
        result = collection.find_one({"fileId": file_id})
        if result:
            return CrawledDocument.from_mongo_dict(result)
        return None
    
    def get_by_source_url(self, source_url: str) -> Optional[CrawledDocument]:
        """Get document by source URL (check for duplicates)."""
        collection = self._get_collection()
        result = collection.find_one({"sourceUrl": source_url, "isDeleted": False})
        if result:
            return CrawledDocument.from_mongo_dict(result)
        return None
    
    def get_pending_documents(self, limit: int = 10) -> List[CrawledDocument]:
        """
        Get documents pending processing.
        
        Args:
            limit: Maximum documents to return
            
        Returns:
            List of pending documents
        """
        collection = self._get_collection()
        cursor = collection.find(
            {"status": DocumentStatus.PENDING.value, "isDeleted": False}
        ).sort("createdAt", ASCENDING).limit(limit)
        
        return [CrawledDocument.from_mongo_dict(doc) for doc in cursor]
    
    def get_by_session(self, crawl_session_id: str) -> List[CrawledDocument]:
        """Get all documents from a crawl session."""
        collection = self._get_collection()
        cursor = collection.find(
            {"crawlSessionId": crawl_session_id, "isDeleted": False}
        ).sort("createdAt", ASCENDING)
        
        return [CrawledDocument.from_mongo_dict(doc) for doc in cursor]
    
    def update_document(self, file_id: str, update_data: dict) -> bool:
        """
        Update document with arbitrary fields.
        
        Args:
            file_id: Document file ID
            update_data: Dictionary of fields to update
            
        Returns:
            True if updated, False if not found
        """
        collection = self._get_collection()
        
        # Always update the updatedAt timestamp
        update_data["updatedAt"] = datetime.utcnow()
        
        result = collection.update_one(
            {"fileId": file_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            logger.debug(f"Updated document {file_id} with {len(update_data)} fields")
            return True
        
        logger.warning(f"Document {file_id} not found for update")
        return False
    
    def update_status(
        self,
        file_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        vector_count: int = 0,
        page_count: int = 0,
        updated_by: str = "system",
    ) -> bool:
        """
        Update document processing status.
        
        Args:
            file_id: Document file ID
            status: New status
            error_message: Error message if failed
            vector_count: Number of vectors stored
            page_count: Number of pages processed
            updated_by: Who made the update
            
        Returns:
            True if updated, False if not found
        """
        collection = self._get_collection()
        
        update_data = {
            "status": status.value,
            "updatedAt": datetime.utcnow(),
            "updatedBy": updated_by,
        }
        
        if error_message is not None:
            update_data["errorMessage"] = error_message
        if vector_count > 0:
            update_data["vectorCount"] = vector_count
        if page_count > 0:
            update_data["pageCount"] = page_count
        
        result = collection.update_one(
            {"fileId": file_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated document {file_id} status to {status.value}")
            return True
        return False
    
    def mark_processing(self, file_id: str) -> bool:
        """Mark document as being processed."""
        return self.update_status(file_id, DocumentStatus.PROCESSING, updated_by="worker")
    
    def mark_vectorized(
        self, file_id: str, vector_count: int, page_count: int = 0
    ) -> bool:
        """Mark document as successfully vectorized."""
        return self.update_status(
            file_id,
            DocumentStatus.VECTORIZED,
            vector_count=vector_count,
            page_count=page_count,
            updated_by="worker",
        )
    
    def mark_failed(self, file_id: str, error_message: str) -> bool:
        """Mark document as failed."""
        return self.update_status(
            file_id,
            DocumentStatus.FAILED,
            error_message=error_message,
            updated_by="worker",
        )
    
    def soft_delete(self, file_id: str, deleted_by: str = "admin") -> bool:
        """Soft delete a document."""
        collection = self._get_collection()
        result = collection.update_one(
            {"fileId": file_id},
            {"$set": {
                "isDeleted": True,
                "updatedAt": datetime.utcnow(),
                "updatedBy": deleted_by,
            }}
        )
        return result.modified_count > 0
    
    def get_stats(self, crawl_session_id: Optional[str] = None) -> dict:
        """Get document statistics."""
        collection = self._get_collection()
        
        match_filter = {"isDeleted": False}
        if crawl_session_id:
            match_filter["crawlSessionId"] = crawl_session_id
        
        pipeline = [
            {"$match": match_filter},
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "totalVectors": {"$sum": "$vectorCount"},
            }}
        ]
        
        results = list(collection.aggregate(pipeline))
        
        stats = {
            "total": 0,
            "pending": 0,
            "processing": 0,
            "vectorized": 0,
            "failed": 0,
            "stored": 0,
            "total_vectors": 0,
        }
    # ============== Nested Website Structure Methods ==============
    
    def create_or_get_website(
        self,
        website_url: str,
        crawl_session_id: str
    ) -> WebsiteCrawl:
        """
        Create a new website document or get existing one.
        
        Args:
            website_url: Base URL of the website
            crawl_session_id: Session ID for this crawl
            
        Returns:
            WebsiteCrawl document
        """
        collection = self._get_websites_collection()
        
        # Try to find existing document
        result = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id
        })
        
        if result:
            return WebsiteCrawl.from_mongo_dict(result)
        
        # Create new document
        website = WebsiteCrawl(
            website_url=website_url,
            crawl_session_id=crawl_session_id
        )
        
        mongo_doc = website.to_mongo_dict()
        collection.insert_one(mongo_doc)
        
        logger.info(f"Created website crawl: {website_url} (session={crawl_session_id})")
        return website
    
    def add_pdf_to_website(
        self,
        website_url: str,
        crawl_session_id: str,
        visited_url: str,
        crawl_depth: int,
        pdf_document: PdfDocument
    ) -> bool:
        """
        Add a PDF to a visited URL within a website.
        Creates the website and visited URL if they don't exist.
        
        Args:
            website_url: Base URL of the website
            crawl_session_id: Session ID for this crawl
            visited_url: The URL where the PDF was found
            crawl_depth: Depth in crawl tree
            pdf_document: The PDF document to add
            
        Returns:
            True if added, False on error
        """
        collection = self._get_websites_collection()
        
        # Check if this URL already has this PDF
        existing = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id,
            "visitedUrls.url": visited_url,
            "visitedUrls.pdfs.fileId": pdf_document.file_id
        })
        
        if existing:
            logger.warning(f"PDF {pdf_document.file_id} already exists in {visited_url}")
            return False
        
        # Check if the visited URL exists
        existing_url = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id,
            "visitedUrls.url": visited_url
        })
        
        if existing_url:
            # Add PDF to existing URL
            result = collection.update_one(
                {
                    "websiteUrl": website_url,
                    "crawlSessionId": crawl_session_id,
                    "visitedUrls.url": visited_url
                },
                {
                    "$push": {"visitedUrls.$.pdfs": pdf_document.to_mongo_dict()},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
        else:
            # Create new visited URL with the PDF
            visited_url_obj = VisitedUrl(
                url=visited_url,
                crawl_depth=crawl_depth,
                pdfs=[pdf_document]
            )
            
            # Check if website exists
            website_exists = collection.find_one({
                "websiteUrl": website_url,
                "crawlSessionId": crawl_session_id
            })
            
            if website_exists:
                # Add visited URL to existing website
                result = collection.update_one(
                    {
                        "websiteUrl": website_url,
                        "crawlSessionId": crawl_session_id
                    },
                    {
                        "$push": {"visitedUrls": visited_url_obj.to_mongo_dict()},
                        "$set": {"updatedAt": datetime.utcnow()}
                    }
                )
            else:
                # Create new website with visited URL
                website = WebsiteCrawl(
                    website_url=website_url,
                    crawl_session_id=crawl_session_id,
                    visited_urls=[visited_url_obj]
                )
                collection.insert_one(website.to_mongo_dict())
                result = type('obj', (object,), {'modified_count': 1})()
        
        if result.modified_count > 0 or not existing_url:
            logger.info(f"Added PDF {pdf_document.file_id} to {visited_url}")
            
            # Automatically update website-level is_crawled and is_visited flags
            self._update_website_crawl_status(website_url, crawl_session_id)
            
            return True
        
        return False
    
    def add_page_to_website(
        self,
        website_url: str,
        crawl_session_id: str,
        visited_url: str,
        crawl_depth: int,
        page_document: "PageDocument"
    ) -> bool:
        """
        Add a page (HTML or PDF) to a visited URL within a website.
        Creates the website and visited URL if they don't exist.
        
        Args:
            website_url: Base URL of the website
            crawl_session_id: Session ID for this crawl
            visited_url: The URL of the page
            crawl_depth: Depth in crawl tree
            page_document: The page document to add (HTML or PDF)
            
        Returns:
            True if added, False on error
        """
        collection = self._get_websites_collection()
        
        # Check if this URL already has this document
        existing = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id,
            "visitedUrls.url": visited_url,
            "visitedUrls.documents.fileId": page_document.file_id
        })
        
        if existing:
            logger.debug(f"Document {page_document.file_id} already exists in {visited_url}")
            # Update existing document instead of returning False
            result = collection.update_one(
                {
                    "websiteUrl": website_url,
                    "crawlSessionId": crawl_session_id,
                    "visitedUrls.url": visited_url,
                    "visitedUrls.documents.fileId": page_document.file_id
                },
                {
                    "$set": {
                        "visitedUrls.$[url].documents.$[doc]": page_document.to_mongo_dict(),
                        "updatedAt": datetime.utcnow()
                    }
                },
                array_filters=[
                    {"url.url": visited_url},
                    {"doc.fileId": page_document.file_id}
                ]
            )
            
            # Auto-update website status after updating existing document
            self._update_website_crawl_status(website_url, crawl_session_id)
            
            return True
        
        # Check if the visited URL exists
        existing_url = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id,
            "visitedUrls.url": visited_url
        })
        
        if existing_url:
            # Add document to existing URL
            result = collection.update_one(
                {
                    "websiteUrl": website_url,
                    "crawlSessionId": crawl_session_id,
                    "visitedUrls.url": visited_url
                },
                {
                    "$push": {"visitedUrls.$.documents": page_document.to_mongo_dict()},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
        else:
            # Create new visited URL with the document
            from app.schemas.document import VisitedUrl
            visited_url_obj = VisitedUrl(
                url=visited_url,
                crawl_depth=crawl_depth,
                documents=[page_document]
            )
            
            # Check if website exists
            website_exists = collection.find_one({
                "websiteUrl": website_url,
                "crawlSessionId": crawl_session_id
            })
            
            if website_exists:
                # Add visited URL to existing website
                result = collection.update_one(
                    {
                        "websiteUrl": website_url,
                        "crawlSessionId": crawl_session_id
                    },
                    {
                        "$push": {"visitedUrls": visited_url_obj.to_mongo_dict()},
                        "$set": {"updatedAt": datetime.utcnow()}
                    }
                )
            else:
                # Create new website with visited URL
                from app.schemas.document import WebsiteCrawl
                website = WebsiteCrawl(
                    website_url=website_url,
                    crawl_session_id=crawl_session_id,
                    visited_urls=[visited_url_obj]
                )
                collection.insert_one(website.to_mongo_dict())
                result = type('obj', (object,), {'modified_count': 1})()
        
        if result.modified_count > 0 or not existing_url:
            logger.info(
                f"Added {page_document.document_type.upper()} document {page_document.file_id} "
                f"to {visited_url} in website {website_url}"
            )
            
            # Automatically update website-level is_crawled and is_visited flags
            self._update_website_crawl_status(website_url, crawl_session_id)
            
            return True
        
        return False
    
    def _update_website_crawl_status(
        self,
        website_url: str,
        crawl_session_id: str
    ) -> None:
        """
        Update website-level is_visited and is_crawled flags based on nested documents.
        Called automatically after adding pages.
        
        Args:
            website_url: Base URL of the website
            crawl_session_id: Session ID for this crawl
        """
        collection = self._get_websites_collection()
        
        # Get the website document
        website_doc = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id
        })
        
        if not website_doc:
            return
        
        # Convert to WebsiteCrawl object and check status
        from app.schemas.document import WebsiteCrawl
        website = WebsiteCrawl.from_mongo_dict(website_doc)
        is_visited, is_crawled = website.check_crawl_status()
        
        # Update in MongoDB if changed
        if website.is_visited != is_visited or website.is_crawled != is_crawled:
            collection.update_one(
                {
                    "websiteUrl": website_url,
                    "crawlSessionId": crawl_session_id
                },
                {
                    "$set": {
                        "isVisited": is_visited,
                        "isCrawled": is_crawled,
                        "updatedAt": datetime.utcnow()
                    }
                }
            )
            logger.info(
                f"Updated website status: {website_url} "
                f"(isVisited={is_visited}, isCrawled={is_crawled})"
            )
    
    def is_url_already_crawled(self, url: str) -> tuple[bool, Optional[dict]]:
        """
        Check if a URL has already been crawled at any depth level.
        
        Searches for the URL in:
        - websiteUrl (top level)
        - visitedUrls.url (nested)
        - visitedUrls.documents.sourceUrl (deep nested)
        
        Args:
            url: The URL to check
            
        Returns:
            Tuple of (is_crawled, details):
            - is_crawled: True if URL exists and has crawled documents
            - details: Dict with crawl info (session_id, website_url, crawl_date) or None
        """
        from urllib.parse import urlparse
        
        collection = self._get_websites_collection()
        
        # Normalize URL for comparison (remove trailing slash, fragments)
        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Search in multiple locations:
        # 1. Check if it's the website base URL
        # 2. Check if it's in visitedUrls array
        # 3. Check if it's in documents.sourceUrl
        
        # First, try to find any website that contains this URL
        query = {
            "$or": [
                {"websiteUrl": {"$in": [url, normalized_url, base_url]}},
                {"visitedUrls.url": {"$in": [url, normalized_url]}},
                {"visitedUrls.documents.sourceUrl": {"$in": [url, normalized_url]}}
            ]
        }
        
        website_docs = list(collection.find(query))
        
        if not website_docs:
            return False, None
        
        # Check if any of the found websites have crawled documents
        for website_doc in website_docs:
            website_url = website_doc.get("websiteUrl", "")
            crawl_session_id = website_doc.get("crawlSessionId", "")
            visited_urls = website_doc.get("visitedUrls", [])
            
            # Check if this specific URL or base URL is crawled
            for visited in visited_urls:
                visited_url = visited.get("url", "")
                
                # Check if this is the URL we're looking for
                if visited_url in [url, normalized_url] or website_url in [url, normalized_url, base_url]:
                    documents = visited.get("documents", [])
                    
                    # If there are documents and at least one is crawled
                    if documents:
                        crawled_docs = [d for d in documents if d.get("isCrawled") == "1"]
                        if crawled_docs:
                            return True, {
                                "website_url": website_url,
                                "crawl_session_id": crawl_session_id,
                                "visited_url": visited_url,
                                "crawled_at": website_doc.get("updatedAt"),
                                "total_documents": len(documents),
                                "crawled_documents": len(crawled_docs),
                                "is_fully_crawled": website_doc.get("isCrawled") == "1"
                            }
            
            # Also check website-level status
            if website_doc.get("isCrawled") == "1" and website_url in [url, normalized_url, base_url]:
                return True, {
                    "website_url": website_url,
                    "crawl_session_id": crawl_session_id,
                    "visited_url": website_url,
                    "crawled_at": website_doc.get("updatedAt"),
                    "total_urls": len(visited_urls),
                    "is_fully_crawled": True
                }
        
        return False, None

    def get_website_by_session(self, crawl_session_id: str) -> List[WebsiteCrawl]:
        """Get all websites from a crawl session."""
        collection = self._get_websites_collection()
        cursor = collection.find({"crawlSessionId": crawl_session_id})
        return [WebsiteCrawl.from_mongo_dict(doc) for doc in cursor]
    
    def get_website(self, website_url: str, crawl_session_id: str) -> Optional[WebsiteCrawl]:
        """Get a specific website crawl."""
        collection = self._get_websites_collection()
        result = collection.find_one({
            "websiteUrl": website_url,
            "crawlSessionId": crawl_session_id
        })
        if result:
            return WebsiteCrawl.from_mongo_dict(result)
        return None
    
    def get_all_pdfs_from_website(
        self,
        website_url: str,
        crawl_session_id: str
    ) -> List[PdfDocument]:
        """Get all PDFs from a website across all visited URLs."""
        website = self.get_website(website_url, crawl_session_id)
        if not website:
            return []
        
        pdfs = []
        for visited_url in website.visited_urls:
            pdfs.extend(visited_url.pdfs)
        return pdfs
    
        
        for r in results:
            status = r["_id"]
            count = r["count"]
            stats["total"] += count
            stats[status] = count
            if status == "vectorized":
                stats["total_vectors"] = r["totalVectors"]
        
        return stats
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if DocumentStore._client:
            DocumentStore._client.close()
            DocumentStore._client = None
            DocumentStore._instance = None
            logger.info("Closed MongoDB connection")
