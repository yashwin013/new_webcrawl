"""
Orchestrator Configuration

Defines worker counts, queue sizes, and resource limits for the multi-site crawler.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerConfig:
    """
    Configuration for worker pools.
    
    IMPORTANT - Memory Considerations for Multiple Websites:
    - Each Docling PDF worker loads ML models: ~3-4GB RAM each
    - pdf_workers=1: Safe for 16GB RAM (10-12GB total usage)
    - pdf_workers=2: Needs 20-24GB RAM total - will cause OOM on 16GB!
    
    For parallel multi-website processing with 16GB RAM:
    - Keep pdf_workers at 1
    - Increase crawler_workers and processor_workers instead
    - Monitor RAM usage to stay under 80% (~13GB of 16GB)
    """
    
    # CPU-bound workers (I/O heavy, can run many for multiple websites)
    # Increased for better parallelism across multiple sites
    crawler_workers: int = 6  # More crawlers for parallel websites (up from 4)
    processor_workers: int = 8  # More processors for concurrent text processing (up from 5)
    storage_workers: int = 3  # Storage writes to Qdrant (up from 2)
    
    # GPU-bound workers (GPU bottleneck, keep minimal)
    pdf_workers: int = 1  # Keep at 1 for 16GB RAM - each Docling worker uses 3-4GB
    ocr_workers: int = 0  # Disabled by default - process separately
    
    @property
    def total_workers(self) -> int:
        """Total number of workers."""
        return (
            self.crawler_workers +
            self.processor_workers +
            self.pdf_workers +
            self.ocr_workers +
            self.storage_workers
        )


@dataclass
class QueueConfig:
    """Configuration for queue sizes (backpressure control)."""
    
    # Website URL queue
    crawl_queue_size: int = 20  # URLs queued for crawling (multiple websites)
    
    # Raw pages waiting for processing (text extraction, chunking)
    processing_queue_size: int = 80  # Pages waiting for text processing
    
    # PDFs waiting for Docling processing (GPU bottleneck - monitor this!)
    pdf_queue_size: int = 30  # Increased from 10 - was at 100% causing blocking
    
    # Pages waiting for OCR (GPU bottleneck - keep reasonable)
    ocr_queue_size: int = 10
    
    # Processed chunks waiting for storage
    storage_queue_size: int = 150  # Large buffer for vector storage writes
    
    @property
    def total_queue_capacity(self) -> int:
        """Total capacity across all queues."""
        return (
            self.crawl_queue_size +
            self.processing_queue_size +
            self.ocr_queue_size +
            self.storage_queue_size
        )


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    
    # CPU limits
    max_cpu_percent: float = 80.0  # Target max CPU usage
    
    # GPU limits
    max_gpu_memory_mb: int = 8192  # Max GPU memory (8GB)
    
    # Memory limits
    max_ram_mb: int = 16384  # Max RAM usage (16GB)
    
    # Rate limiting
    max_requests_per_second: float = 10.0  # Per website
    max_concurrent_requests: int = 5  # Per website


@dataclass
class RecoveryConfig:
    """
    Configuration for worker recovery system.
    
    Timeouts increased for large PDF processing which can take several minutes.
    Workers exceeding timeout are killed and restarted to prevent hung processes.
    """
    
    # Enable automatic worker recovery
    enable_recovery: bool = True
    
    # How often to check worker health (seconds)
    check_interval: float = 15.0
    
    # Maximum retries before abandoning a task (reduced to fail fast)
    max_task_retries: int = 1
    
    # Worker-specific timeouts (seconds) - optimized for performance
    # Reduced timeouts to fail fast on problematic documents
    crawler_timeout: float = 240.0  # 4 minutes for large sites
    processor_timeout: float = 180.0  # 3 minutes (fail fast on stuck pages)
    pdf_timeout: float = 120.0  # 2 minutes for PDFs (6 min was too long)
    ocr_timeout: float = 180.0  # 3 minutes
    storage_timeout: float = 60.0  # 1 minute


@dataclass
class OrchestratorConfig:
    """Complete orchestrator configuration."""
    
    workers: WorkerConfig
    queues: QueueConfig
    limits: ResourceLimits
    recovery: RecoveryConfig
    
    # Monitoring (reduced interval for performance)
    enable_monitoring: bool = True
    monitoring_interval_seconds: float = 10.0  # Up from 5s to reduce overhead
    
    # Graceful shutdown
    shutdown_timeout_seconds: float = 30.0
    
    # Progress tracking
    save_progress_interval_seconds: float = 10.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.workers.ocr_workers > 2:
            raise ValueError(
                "ocr_workers should not exceed 2 to prevent GPU overload. "
                f"Got: {self.workers.ocr_workers}"
            )
        
        if self.queues.ocr_queue_size > 20:
            raise ValueError(
                "ocr_queue_size should not exceed 20 to prevent memory issues. "
                f"Got: {self.queues.ocr_queue_size}"
            )
    
    @property
    def summary(self) -> str:
        """Human-readable configuration summary."""
        return f"""
Orchestrator Configuration:
  Workers: {self.workers.total_workers} total
    - Crawlers: {self.workers.crawler_workers} (CPU)
    - Processors: {self.workers.processor_workers} (CPU)
    - PDF: {self.workers.pdf_workers} (GPU)
    - OCR: {self.workers.ocr_workers} (GPU)
    - Storage: {self.workers.storage_workers} (CPU)
  
  Queue Capacity: {self.queues.total_queue_capacity} items
    - Crawl: {self.queues.crawl_queue_size}
    - Processing: {self.queues.processing_queue_size}
    - PDF: {self.queues.pdf_queue_size}
    - OCR: {self.queues.ocr_queue_size}
    - Storage: {self.queues.storage_queue_size}
  
  Resource Limits:
    - CPU: {self.limits.max_cpu_percent}%
    - GPU Memory: {self.limits.max_gpu_memory_mb}MB
    - RAM: {self.limits.max_ram_mb}MB
    - Requests/sec: {self.limits.max_requests_per_second}
  
  Recovery: {"Enabled" if self.recovery.enable_recovery else "Disabled"}
    - Check Interval: {self.recovery.check_interval}s
    - Max Retries: {self.recovery.max_task_retries}
    - PDF Timeout: {self.recovery.pdf_timeout}s
"""


def get_default_config() -> OrchestratorConfig:
    """Get default configuration optimized for performance."""
    return OrchestratorConfig(
        workers=WorkerConfig(
            crawler_workers=6,  # Increased for faster parallel crawling
            processor_workers=8,  # Increased for better text processing throughput
            pdf_workers=1,  # Keep at 1 for 16GB RAM (each worker = 3-4GB)
            ocr_workers=0,  # Disabled - OCR backlog processed separately
            storage_workers=3,  # Increased for faster database writes
        ),
        queues=QueueConfig(
            crawl_queue_size=20,  # Double size for multiple websites
            processing_queue_size=80,  # More buffer for parallel processing
            pdf_queue_size=30,  # Critical: was at 100%, increased to 30
            ocr_queue_size=10,
            storage_queue_size=150,  # More buffer for chunks
        ),
        limits=ResourceLimits(
            max_cpu_percent=80.0,
            max_gpu_memory_mb=8192,
            max_ram_mb=16384,
            max_requests_per_second=10.0,
            max_concurrent_requests=5,
        ),
        recovery=RecoveryConfig(
            enable_recovery=True,
            check_interval=15.0,
            max_task_retries=1,  # Reduced from 2 - fail fast on problematic pages
            crawler_timeout=240.0,  # Reduced from 300s
            processor_timeout=180.0,  # Reduced from 300s - fail fast
            pdf_timeout=120.0,  # Reduced from 360s - major speed improvement
            ocr_timeout=180.0,  # Reduced from 300s
            storage_timeout=60.0,  # Reduced from 120s
        ),
        enable_monitoring=True,
        monitoring_interval_seconds=10.0,  # Increased from 5s to reduce overhead,
        shutdown_timeout_seconds=30.0,
        save_progress_interval_seconds=10.0,
    )


def get_light_config() -> OrchestratorConfig:
    """Get lightweight configuration for low-resource systems."""
    return OrchestratorConfig(
        workers=WorkerConfig(
            crawler_workers=2,
            processor_workers=2,
            pdf_workers=1,
            ocr_workers=0,
            storage_workers=1,
        ),
        queues=QueueConfig(
            crawl_queue_size=5,
            processing_queue_size=20,
            pdf_queue_size=5,
            ocr_queue_size=5,
            storage_queue_size=30,
        ),
        limits=ResourceLimits(
            max_cpu_percent=60.0,
            max_gpu_memory_mb=4096,
            max_ram_mb=8192,
            max_requests_per_second=5.0,
            max_concurrent_requests=3,
        ),
        recovery=RecoveryConfig(
            enable_recovery=True,
            check_interval=20.0,
            max_task_retries=1,
            crawler_timeout=240.0,
            processor_timeout=150.0,
            pdf_timeout=200.0,
            ocr_timeout=240.0,
            storage_timeout=100.0,
        ),
        enable_monitoring=True,
        monitoring_interval_seconds=10.0,
        shutdown_timeout_seconds=20.0,
        save_progress_interval_seconds=15.0,
    )


def get_aggressive_config() -> OrchestratorConfig:
    """Get aggressive configuration for high-resource systems."""
    return OrchestratorConfig(
        workers=WorkerConfig(
            crawler_workers=5,
            processor_workers=6,
            pdf_workers=2,
            ocr_workers=1,
            storage_workers=3,
        ),
        queues=QueueConfig(
            crawl_queue_size=20,
            processing_queue_size=100,
            pdf_queue_size=15,
            ocr_queue_size=15,
            storage_queue_size=200,
        ),
        limits=ResourceLimits(
            max_cpu_percent=90.0,
            max_gpu_memory_mb=16384,
            max_ram_mb=32768,
            max_requests_per_second=20.0,
            max_concurrent_requests=10,
        ),
        recovery=RecoveryConfig(
            enable_recovery=True,
            check_interval=10.0,
            max_task_retries=3,
            crawler_timeout=360.0,
            processor_timeout=240.0,
            pdf_timeout=300.0,
            ocr_timeout=360.0,
            storage_timeout=150.0,
        ),
        enable_monitoring=True,
        monitoring_interval_seconds=3.0,
        shutdown_timeout_seconds=45.0,
        save_progress_interval_seconds=5.0,
    )
