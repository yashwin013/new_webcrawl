"""
Real-time Resource Monitor for Multi-Website Crawling

Monitors CPU, RAM, and GPU usage while processing multiple websites.
Run this in a separate terminal to track resource usage.
"""

import psutil
import time
import sys
from datetime import datetime


def format_bytes(bytes_val):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def get_gpu_info():
    """Try to get GPU memory usage (requires nvidia-smi or similar)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            return int(used.strip()), int(total.strip())
    except:
        pass
    return None, None


def monitor_loop(interval=5, max_iterations=None):
    """Monitor system resources in a loop."""
    print("=" * 70)
    print("Resource Monitor for Multi-Website Crawling")
    print("=" * 70)
    print(f"Monitoring every {interval} seconds. Press Ctrl+C to stop.\n")
    
    iteration = 0
    max_ram_percent = 0
    max_ram_gb = 0
    
    try:
        while True:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                break
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # RAM Usage
            memory = psutil.virtual_memory()
            ram_used_gb = memory.used / (1024**3)
            ram_total_gb = memory.total / (1024**3)
            ram_percent = memory.percent
            
            # Track maximums
            if ram_percent > max_ram_percent:
                max_ram_percent = ram_percent
                max_ram_gb = ram_used_gb
            
            # GPU Usage (if available)
            gpu_used, gpu_total = get_gpu_info()
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Display
            print(f"\r[{timestamp}] ", end="")
            print(f"CPU: {cpu_percent:5.1f}% | ", end="")
            print(f"RAM: {ram_used_gb:5.1f}/{ram_total_gb:.1f}GB ({ram_percent:5.1f}%) | ", end="")
            
            if gpu_used is not None:
                gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
                print(f"GPU: {gpu_used}/{gpu_total}MB ({gpu_percent:.1f}%)", end="")
            
            # Warning if RAM is high
            if ram_percent > 85:
                print("  ⚠️  HIGH RAM!", end="")
            
            sys.stdout.flush()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Monitoring stopped.")
        print(f"Peak RAM Usage: {max_ram_gb:.1f}GB ({max_ram_percent:.1f}%)")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor system resources for crawling")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    args = parser.parse_args()
    
    monitor_loop(interval=args.interval)
