"""
Health check system for production monitoring.

Provides health check endpoints and status monitoring for all system components.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from app.config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """
    Health check system for monitoring application components.
    
    Usage:
        health = HealthCheck()
        
        # Register component checks
        health.register_check("mongodb", check_mongodb_connection)
        health.register_check("qdrant", check_qdrant_connection)
        
        # Get overall health
        status = await health.check()
    """
    
    def __init__(self):
        """Initialize health check system."""
        self._checks: Dict[str, callable] = {}
        self._last_check: Optional[datetime] = None
        self._cached_result: Optional[Dict[str, Any]] = None
        self._cache_ttl: float = 5.0  # Cache results for 5 seconds
    
    def register_check(self, name: str, check_func: callable) -> None:
        """
        Register a health check function.
        
        Args:
            name: Component name (e.g., "mongodb", "qdrant")
            check_func: Async function that returns dict with 'status' key
        """
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    async def check(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run all health checks and return overall status.
        
        Args:
            use_cache: Use cached results if recent enough
            
        Returns:
            Dict with overall status and component details
        """
        # Return cached result if valid
        if use_cache and self._is_cache_valid():
            return self._cached_result
        
        results = {}
        unhealthy_count = 0
        degraded_count = 0
        
        # Run all checks in parallel
        check_tasks = []
        check_names = []
        
        for name, check_func in self._checks.items():
            check_tasks.append(self._run_check(name, check_func))
            check_names.append(name)
        
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        for name, result in zip(check_names, check_results):
            if isinstance(result, Exception):
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                }
                unhealthy_count += 1
            else:
                results[name] = result
                status = result.get("status", HealthStatus.HEALTHY.value)
                if status == HealthStatus.UNHEALTHY.value:
                    unhealthy_count += 1
                elif status == HealthStatus.DEGRADED.value:
                    degraded_count += 1
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Build response
        response = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": results,
            "summary": {
                "total": len(results),
                "healthy": len(results) - unhealthy_count - degraded_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            }
        }
        
        # Cache the result
        self._cached_result = response
        self._last_check = datetime.utcnow()
        
        return response
    
    async def _run_check(self, name: str, check_func: callable) -> Dict[str, Any]:
        """Run a single health check with timeout."""
        try:
            # Run check with 5 second timeout
            result = await asyncio.wait_for(check_func(), timeout=5.0)
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {"status": HealthStatus.HEALTHY.value}
            if "status" not in result:
                result["status"] = HealthStatus.HEALTHY.value
            if "timestamp" not in result:
                result["timestamp"] = datetime.utcnow().isoformat()
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout: {name}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "error": "Health check timed out after 5s",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {name} - {e}")
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached result is still valid."""
        if self._last_check is None or self._cached_result is None:
            return False
        
        elapsed = (datetime.utcnow() - self._last_check).total_seconds()
        return elapsed < self._cache_ttl
    
    async def check_component(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Check health of a specific component.
        
        Args:
            name: Component name
            
        Returns:
            Component health status or None if not registered
        """
        check_func = self._checks.get(name)
        if not check_func:
            return None
        
        return await self._run_check(name, check_func)


# Global health check instance
_health_check_instance: Optional[HealthCheck] = None


def get_health_check() -> HealthCheck:
    """Get global health check instance."""
    global _health_check_instance
    if _health_check_instance is None:
        _health_check_instance = HealthCheck()
    return _health_check_instance


# Common health check functions
async def check_mongodb_health() -> Dict[str, Any]:
    """Check MongoDB connection health."""
    try:
        from app.core.database import db_manager
        
        # Ensure connection is established
        if not db_manager.is_connected:
            try:
                await db_manager.connect()
            except Exception as conn_error:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Failed to connect to MongoDB",
                    "error": str(conn_error)
                }
        
        # Try a simple ping command
        db = db_manager.get_database()
        await db.command('ping')
        
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "MongoDB connection healthy"
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "error": str(e)
        }


async def check_qdrant_health() -> Dict[str, Any]:
    """Check Qdrant connection health."""
    try:
        from app.config import get_qdrant_client
        import asyncio
        
        # Get client and check collections (blocking operation)
        client = get_qdrant_client()
        collections = await asyncio.to_thread(client.get_collections)
        
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Qdrant connection healthy",
            "details": {
                "collections": len(collections.collections)
            }
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "error": str(e)
        }
