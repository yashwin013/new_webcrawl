#!/usr/bin/env python3
"""
Health Check CLI Tool

Check the health status of all system components.

Usage:
    python cli/check_health.py           # Check all components
    python cli/check_health.py --json    # Output as JSON
    python cli/check_health.py --watch   # Continuous monitoring mode
"""

import asyncio
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.health_registry import register_health_checks, get_full_health_status
from app.config import get_logger

logger = get_logger(__name__)


def print_health_status(status: dict, json_output: bool = False):
    """Print health status in human-readable or JSON format."""
    if json_output:
        print(json.dumps(status, indent=2))
        return
    
    # Human-readable output
    overall = status.get('status', 'unknown')
    overall_icon = "✓" if overall == "healthy" else "✗"
    
    print()
    print("=" * 70)
    print(f"SYSTEM HEALTH CHECK - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n{overall_icon} Overall Status: {overall.upper()}\n")
    print("Component Status:")
    print("-" * 70)
    
    for name, result in status['components'].items():
        component_status = result.get('status', 'unknown')
        status_icon = "✓" if component_status == 'healthy' else "✗"
        
        # Format component name (capitalize, pad to 12 chars)
        component_display = name.capitalize().ljust(12)
        
        print(f"{status_icon} {component_display} {component_status.upper()}")
        
        # Show error if present
        if 'error' in result:
            print(f"  └─ Error: {result['error']}")
        
        # Show details if present
        if 'details' in result:
            for key, value in result['details'].items():
                print(f"  └─ {key}: {value}")
        
        # Show message if present
        if 'message' in result and component_status != 'healthy':
            print(f"  └─ {result['message']}")
        
        # GPU-specific details
        if name == 'gpu' and component_status == 'healthy':
            if 'device_count' in result:
                print(f"  └─ Devices: {result['device_count']}")
            if 'device_name' in result and result['device_name']:
                print(f"  └─ Device: {result['device_name']}")
            if 'memory_allocated_gb' in result:
                print(f"  └─ Memory: {result['memory_allocated_gb']} GB allocated")
    
    print("-" * 70)
    print()


async def check_once(json_output: bool = False):
    """Run health check once and print results."""
    try:
        # Register health checks
        register_health_checks()
        
        # Get health status
        status = await get_full_health_status()
        
        # Print results
        print_health_status(status, json_output)
        
        # Exit code based on overall health
        sys.exit(0 if status.get('status') == 'healthy' else 1)
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        if json_output:
            print(json.dumps({"error": str(e), "overall": "error"}, indent=2))
        else:
            print(f"\n✗ Health check failed: {e}\n")
        sys.exit(2)


async def watch_mode(interval: int = 10):
    """Continuously monitor health status."""
    try:
        # Register health checks once
        register_health_checks()
        
        print("\n" + "=" * 70)
        print("CONTINUOUS HEALTH MONITORING")
        print(f"Checking every {interval} seconds (Press Ctrl+C to stop)")
        print("=" * 70 + "\n")
        
        while True:
            try:
                status = await get_full_health_status()
                print_health_status(status, json_output=False)
                
                # Wait before next check
                print(f"Next check in {interval} seconds...")
                await asyncio.sleep(interval)
                
                # Clear screen for cleaner output
                print("\033[H\033[J", end="")  # ANSI escape to clear screen
            
            except KeyboardInterrupt:
                print("\n\n✓ Monitoring stopped by user\n")
                break
    
    except Exception as e:
        logger.error(f"Watch mode failed: {e}", exc_info=True)
        sys.exit(2)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check health status of all system components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/check_health.py              # Single check
  python cli/check_health.py --json       # JSON output
  python cli/check_health.py --watch      # Continuous monitoring
  python cli/check_health.py --watch --interval 30  # Check every 30s
        """
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of human-readable format'
    )
    
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuously monitor health status'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Interval in seconds for watch mode (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.watch:
        asyncio.run(watch_mode(args.interval))
    else:
        asyncio.run(check_once(args.json))


if __name__ == "__main__":
    main()
