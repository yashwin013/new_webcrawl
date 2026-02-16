"""
CLI Tool: Check if URL was already crawled

Usage:
    python check_url.py <url>
    
Example:
    python check_url.py https://example.com
    python check_url.py "https://example.com/docs"
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.document_store import DocumentStore

def check_url(url: str):
    """Check if a URL has been crawled."""
    store = DocumentStore.from_config()
    
    print("="*80)
    print(f"Checking URL: {url}")
    print("="*80)
    
    is_crawled, details = store.is_url_already_crawled(url)
    
    if is_crawled:
        print("\n✅ STATUS: ALREADY CRAWLED")
        print("-"*80)
        print(f"Website URL:     {details['website_url']}")
        print(f"Session ID:      {details['crawl_session_id']}")
        print(f"Visited URL:     {details['visited_url']}")
        print(f"Crawled At:      {details.get('crawled_at', 'N/A')}")
        
        if details.get('is_fully_crawled'):
            print(f"Status:          ✓ Fully crawled")
            print(f"Total URLs:      {details.get('total_urls', 'N/A')}")
        else:
            crawled = details.get('crawled_documents', 0)
            total = details.get('total_documents', 0)
            print(f"Status:          ⚠ Partially crawled")
            print(f"Documents:       {crawled}/{total} crawled")
        
        print("-"*80)
        print("\n💡 Recommendation:")
        print("   This URL will be SKIPPED if you try to crawl it again.")
        print("   The existing crawl data will be used instead.")
        
    else:
        print("\n❌ STATUS: NOT CRAWLED")
        print("-"*80)
        print("This URL has not been crawled yet.")
        print("-"*80)
        print("\n💡 Recommendation:")
        print("   This URL is safe to crawl. It will be processed normally.")
    
    print("="*80)
    return is_crawled

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_url.py <url>")
        print("\nExample:")
        print("  python check_url.py https://example.com")
        sys.exit(1)
    
    url = sys.argv[1]
    is_crawled = check_url(url)
    
    # Exit code 0 = not crawled (safe to crawl)
    # Exit code 1 = already crawled (will be skipped)
    sys.exit(1 if is_crawled else 0)
