"""
Robots.txt parsing and sitemap utilities.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Set
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import requests

from app.config import get_logger

logger = get_logger(__name__)


@dataclass
class RobotsRules:
    """
    Parsed robots.txt rules.
    
    Usage:
        rules = parse_robots_txt("https://example.com")
        
        if rules.can_fetch("/some/path"):
            # OK to crawl
            pass
            
        delay = rules.get_crawl_delay()
    """
    parser: RobotFileParser
    crawl_delay: float = 0.0
    
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched."""
        return self.parser.can_fetch(user_agent, url)
    
    def get_crawl_delay(self) -> float:
        """Get crawl delay from robots.txt."""
        return self.crawl_delay


def parse_robots_txt(base_url: str, user_agent: str = "*") -> RobotsRules:
    """
    Parse robots.txt and return rules.
    
    Args:
        base_url: Base URL of the website
        user_agent: User agent to check rules for
        
    Returns:
        RobotsRules object
    """
    parser = RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    
    try:
        parser.set_url(robots_url)
        parser.read()
        
        # Get crawl delay
        crawl_delay = 0.0
        try:
            delay = parser.crawl_delay(user_agent)
            if delay:
                crawl_delay = float(delay)
        except Exception:
            pass
        
        # Check if parser actually loaded rules (has content)
        # If robots.txt has no User-agent rules, allow everything
        if not hasattr(parser, 'entries') or not parser.entries:
            logger.info(f"No robot rules found in {robots_url}, allowing all")
            permissive_parser = RobotFileParser()
            permissive_parser.set_url(robots_url)
            # Create a permissive robots.txt that allows everything
            permissive_parser.parse(["User-agent: *", "Allow: /"])
            return RobotsRules(parser=permissive_parser, crawl_delay=crawl_delay)
        
        return RobotsRules(parser=parser, crawl_delay=crawl_delay)
        
    except Exception as e:
        logger.warning(f"Failed to parse robots.txt at {robots_url}: {e}")
        # Return permissive rules on failure - allow everything
        permissive_parser = RobotFileParser()
        permissive_parser.set_url(robots_url)
        permissive_parser.parse(["User-agent: *", "Allow: /"])
        return RobotsRules(parser=permissive_parser, crawl_delay=0.0)


def parse_sitemap(base_url: str, timeout: int = 30) -> Set[str]:
    """
    Parse sitemap.xml to discover all pages.
    
    Supports:
    - Standard sitemap.xml
    - Sitemap index files (nested sitemaps)
    - Compressed sitemaps (.gz)
    
    Args:
        base_url: Base URL of the website
        timeout: Request timeout in seconds
        
    Returns:
        Set of discovered URLs
    """
    discovered_urls: Set[str] = set()
    
    # Try common sitemap locations
    sitemap_locations = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"),
        urljoin(base_url, "/sitemap/sitemap.xml"),
    ]
    
    def fetch_sitemap(sitemap_url: str) -> None:
        """Fetch and parse a single sitemap."""
        try:
            # Use provided timeout or fall back to centralized config
            from app.core.timeouts import TimeoutConfig
            timeout_config = TimeoutConfig()
            actual_timeout = timeout if timeout else timeout_config.CRAWLER_PAGE_LOAD
            
            response = requests.get(sitemap_url, timeout=actual_timeout)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle namespace
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            # Check if this is a sitemap index
            sitemap_refs = root.findall(".//sm:sitemap/sm:loc", ns)
            if sitemap_refs:
                # Recursively fetch nested sitemaps
                for ref in sitemap_refs:
                    if ref.text:
                        fetch_sitemap(ref.text)
            else:
                # Regular sitemap - extract URLs
                urls = root.findall(".//sm:url/sm:loc", ns)
                for url in urls:
                    if url.text:
                        discovered_urls.add(url.text)
                        
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to fetch sitemap {sitemap_url}: {e}")
        except ET.ParseError as e:
            logger.debug(f"Failed to parse sitemap {sitemap_url}: {e}")
    
    # Try each location
    for location in sitemap_locations:
        fetch_sitemap(location)
    
    logger.info(f"Discovered {len(discovered_urls)} URLs from sitemap")
    return discovered_urls
