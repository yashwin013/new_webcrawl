"""
Demo: Viewing Website Crawl Status with Auto-Updating Flags

This script demonstrates how to query the websites collection
and see the is_visited and is_crawled flags in action.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymongo import MongoClient
from app.config import MONGODB_URL
import json

def view_website_status(website_url: str = None, crawl_session_id: str = None):
    """
    View website crawl status from MongoDB.
    
    Args:
        website_url: Optional filter by website URL
        crawl_session_id: Optional filter by session ID
    """
    client = MongoClient(MONGODB_URL)
    db = client["crawl"]
    collection = db["websites"]
    
    # Build query
    query = {}
    if website_url:
        query["websiteUrl"] = website_url
    if crawl_session_id:
        query["crawlSessionId"] = crawl_session_id
    
    # Get all matching websites
    websites = list(collection.find(query).sort("createdAt", -1).limit(10))
    
    print("="*80)
    print(f"WEBSITE CRAWL STATUS REPORT")
    print("="*80)
    
    if not websites:
        print("\nNo websites found with the given filters.")
        client.close()
        return
    
    for idx, website in enumerate(websites, 1):
        print(f"\n{idx}. Website: {website.get('websiteUrl', 'N/A')}")
        print(f"   Session ID: {website.get('crawlSessionId', 'N/A')}")
        print(f"   ┌─ Status Flags:")
        print(f"   │  is_visited: {website.get('isVisited', '0')} {'✓ All URLs visited' if website.get('isVisited') == '1' else '⚠ Not all URLs visited'}")
        print(f"   │  is_crawled: {website.get('isCrawled', '0')} {'✓ All documents crawled' if website.get('isCrawled') == '1' else '⚠ Not all documents crawled'}")
        print(f"   └─ Created: {website.get('createdAt', 'N/A')}")
        
        visited_urls = website.get('visitedUrls', [])
        print(f"   Total URLs: {len(visited_urls)}")
        
        # Show details of each visited URL
        for url_idx, visited_url in enumerate(visited_urls[:5], 1):  # Show first 5 URLs
            url = visited_url.get('url', 'N/A')
            documents = visited_url.get('documents', [])
            print(f"\n   └─ URL {url_idx}: {url}")
            print(f"      Documents: {len(documents)}")
            
            for doc_idx, doc in enumerate(documents[:3], 1):  # Show first 3 docs per URL
                doc_type = doc.get('documentType', 'unknown')
                file_id = doc.get('fileId', 'N/A')
                is_crawled = doc.get('isCrawled', '0')
                is_vectorized = doc.get('isVectorized', '0')
                vector_count = doc.get('vectorCount', 0)
                status = doc.get('status', 'unknown')
                
                crawl_icon = "✓" if is_crawled == "1" else "⏳"
                vector_icon = "✓" if is_vectorized == "1" else "⏳"
                
                print(f"      └─ {doc_type.upper()} {doc_idx}:")
                print(f"         File: {file_id[:20]}...")
                print(f"         Status: {status}")
                print(f"         Crawled: {crawl_icon}")
                print(f"         Vectorized: {vector_icon} ({vector_count} chunks)")
            
            if len(documents) > 3:
                print(f"      ... and {len(documents) - 3} more documents")
        
        if len(visited_urls) > 5:
            print(f"\n   ... and {len(visited_urls) - 5} more URLs")
        
        print("-"*80)
    
    print(f"\nTotal websites shown: {len(websites)}")
    client.close()

if __name__ == "__main__":
    import sys
    
    # Parse command line args
    website_url = sys.argv[1] if len(sys.argv) > 1 else None
    crawl_session_id = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("\nUsage: python view_website_status.py [website_url] [crawl_session_id]")
    print("       Showing latest websites...\n")
    
    view_website_status(website_url, crawl_session_id)
