"""
Health Check Registry

Registers all system health checks for monitoring.
"""

import asyncio
from typing import Dict, Any

from app.core.health import get_health_check, check_mongodb_health, check_qdrant_health
from app.config import get_logger

logger = get_logger(__name__)

# Optional torch import for GPU checks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, GPU health checks will report unavailable")


async def check_gpu_health() -> Dict[str, Any]:
    """Check if GPU is available and responsive."""
    if not TORCH_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "torch module not installed",
            "device_count": 0
        }
    
    try:
        if not torch.cuda.is_available():
            return {
                "status": "unavailable",
                "message": "CUDA not available",
                "device_count": 0
            }
        
        # Simple tensor operation to verify GPU works
        test = torch.zeros(1).cuda()
        del test
        torch.cuda.empty_cache()
        
        device_count = torch.cuda.device_count()
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        return {
            "status": "healthy",
            "device_count": device_count,
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0) if device_count > 0 else None
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_worker_health() -> Dict[str, Any]:
    """
    Check health of orchestrator workers.
    This is a placeholder - implement based on your worker monitoring system.
    """
    try:
        # TODO: Integrate with actual worker monitoring
        # For now, return healthy if system is initialized
        return {
            "status": "healthy",
            "message": "Worker health check not implemented"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def register_health_checks():
    """Register all system health checks."""
    health = get_health_check()
    
    # Register MongoDB
    health.register_check("mongodb", check_mongodb_health)
    logger.info("✓ Registered MongoDB health check")
    
    # Register Qdrant
    health.register_check("qdrant", check_qdrant_health)
    logger.info("✓ Registered Qdrant health check")
    
    # Register GPU
    health.register_check("gpu", check_gpu_health)
    logger.info("✓ Registered GPU health check")
    
    # Register Workers (placeholder)
    health.register_check("workers", check_worker_health)
    logger.info("✓ Registered Workers health check")
    
    logger.info("=" * 60)
    logger.info("All health checks registered successfully")
    logger.info("=" * 60)


async def get_full_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status of all components.
    
    Returns:
        Dict with overall status and component details
    """
    health = get_health_check()
    return await health.check()
