"""
MongoDB Connection Manager with proper lifecycle management.

Provides async context managers and connection pooling for MongoDB operations.
Ensures connections are properly closed and resources are cleaned up.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from app.config import app_config
from app.core.circuit_breaker import CircuitBreaker
from app.core.timeouts import TimeoutConfig

logger = logging.getLogger("database_manager")
timeout_config = TimeoutConfig()


class DatabaseManager:
    """
    Manages MongoDB connections with proper lifecycle.
    
    Features:
    - Connection pooling configuration
    - Automatic connection cleanup
    - Context managers for safe collection access
    - Singleton pattern to prevent multiple clients
    """
    
    _instance: Optional["DatabaseManager"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _connection_lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(self):
        """Initialize database manager (use get_instance() instead)."""
        self._is_connected = False
        # Circuit breaker for MongoDB operations
        self._breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2
        )
        
    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """Get singleton instance of database manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def connect(self) -> None:
        """
        Initialize MongoDB connection with pooling configuration.
        Safe to call multiple times - only connects once.
        """
        async with self._connection_lock:
            if self._is_connected and self._client is not None:
                return
                
            try:
                self._client = AsyncIOMotorClient(
                    app_config.MONGODB_URL,
                    maxPoolSize=50,          # Max connections in pool
                    minPoolSize=10,          # Min connections to maintain
                    maxIdleTimeMS=30000,     # Close idle connections after 30s
                    serverSelectionTimeoutMS=timeout_config.MONGODB_SERVER_SELECTION,
                    connectTimeoutMS=timeout_config.MONGODB_CONNECT,
                    socketTimeoutMS=timeout_config.MONGODB_SOCKET,
                )
                
                # Test connection with circuit breaker
                async with self._breaker:
                    await self._client.admin.command('ping')
                self._is_connected = True
                logger.info(f"✓ Connected to MongoDB: {app_config.MONGODB_DATABASE}")
                logger.info(f"  Pool: {10}-{50} connections, 30s idle timeout")
                
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
                self._client = None
                self._is_connected = False
                raise
    
    async def disconnect(self) -> None:
        """
        Close all MongoDB connections.
        Should be called on application shutdown.
        """
        async with self._connection_lock:
            if self._client is not None:
                try:
                    self._client.close()
                    logger.info("✓ Closed MongoDB connections")
                except Exception as e:
                    logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
                finally:
                    self._client = None
                    self._is_connected = False
    
    def get_database(self, db_name: Optional[str] = None) -> AsyncIOMotorDatabase:
        """
        Get database instance.
        
        Args:
            db_name: Database name (defaults to configured database)
            
        Returns:
            AsyncIOMotorDatabase instance
            
        Raises:
            RuntimeError: If not connected
        """
        if not self._is_connected or self._client is None:
            raise RuntimeError("Database not connected. Call await db_manager.connect() first.")
        
        db_name = db_name or app_config.MONGODB_DATABASE
        return self._client[db_name]
    
    def get_collection(
        self, 
        collection_name: str, 
        db_name: Optional[str] = None
    ) -> AsyncIOMotorCollection:
        """
        Get collection instance.
        
        Args:
            collection_name: Name of collection
            db_name: Database name (defaults to configured database)
            
        Returns:
            AsyncIOMotorCollection instance
            
        Raises:
            RuntimeError: If not connected
        """
        db = self.get_database(db_name)
        return db[collection_name]
    
    @asynccontextmanager
    async def collection_context(
        self, 
        collection_name: str, 
        db_name: Optional[str] = None
    ):
        """
        Context manager for safe collection access with circuit breaker protection.
        
        Usage:
            async with db_manager.collection_context("files") as collection:
                await collection.insert_one({"key": "value"})
                
        Args:
            collection_name: Name of collection
            db_name: Database name (defaults to configured database)
            
        Yields:
            AsyncIOMotorCollection instance
        """
        try:
            async with self._breaker:
                collection = self.get_collection(collection_name, db_name)
                yield collection
        except Exception as e:
            logger.error(
                f"Error in collection context '{collection_name}': {e}", 
                exc_info=True
            )
            raise
        # Connection automatically returned to pool - no explicit cleanup needed
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected and self._client is not None


# Global singleton instance
db_manager = DatabaseManager.get_instance()


# Convenience functions for backward compatibility
async def get_file_collection() -> AsyncIOMotorCollection:
    """
    Get files collection (backward compatible).
    Ensures connection is established.
    """
    if not db_manager.is_connected:
        await db_manager.connect()
    return db_manager.get_collection("files")


async def get_documents_collection() -> AsyncIOMotorCollection:
    """Get documents collection with connection check."""
    if not db_manager.is_connected:
        await db_manager.connect()
    return db_manager.get_collection("documents")


# Application lifecycle hooks
async def startup_database():
    """Call this on application startup."""
    await db_manager.connect()


async def shutdown_database():
    """Call this on application shutdown."""
    await db_manager.disconnect()
