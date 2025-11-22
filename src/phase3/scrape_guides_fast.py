"""Fast parallel scraper using async requests."""
import asyncio
import aiohttp
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import json

from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

from config.settings import GUIDES_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FastGuidesScraper:
    """Fast parallel scraper for LaCuraDellAuto guides."""
    
    BASE_URL = "https://www.lacuradellauto.it"
    GUIDES_PAGE = f"{BASE_URL}/guida-detailing"
    
    def __init__(self, max_concurrent: int = 5, delay: float = 0.2):
        """Initialize fast scraper.
        
        Args:
            max_concurrent: Maximum concurrent requests (default: 5)
            delay: Delay between requests in seconds (default: 0.2)
        """
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.guides = []
        self.session = None
        self.semaphore = None
        
    async def create_session(self):
        """Create aiohttp session."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        self.session = aiohttp.ClientSession(headers=headers)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content asynchronously.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        async with self.semaphore:
            try:
                logger.debug(f"Fetching: {url}")
                
                async with self.session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    content = await response.text()
                    
                    # Small delay to be respectful
                    await asyncio.sleep(self.delay)
                    
                    return content
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
    
    def parse_soup(self, html: str) -> BeautifulSoup:
        """Parse HTML to BeautifulSoup."""
        return BeautifulSoup(html, 'html.parser')
    
    def extract_guide_links(self, html: str) -> List[Dict[str, str]]:
        """Extract all guide links from main page HTML.
        
        Args:
            html: HTML content of main page
            
        Returns:
            List of dicts with guide info
        """
        soup = self.parse_soup(html)
        guides = []
        
        guide_cards = soup.find_all('div', class_='o_colored_level')
        
        for card in guide_cards:
            try:
                guide_number_elem = card.find('b')
                guide_number = guide_number_elem.get_text(strip=True) if guide_number_elem else ""
                
                title_elem = card.find('h2')
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                desc_elem = card.find('p', class_='card-text-1')
                description = desc_elem.get_text(strip=True) if desc_elem else ""
                
                link_elem = card.find('a', class_='btn-primary-guide')
                if link_elem and link_elem.get('href'):
                    guide_url = urljoin(self.BASE_URL, link_elem['href'])
                    
                    guides.append({
                        'guide_number': guide_number,
                        'title': title,
                        'description': description,
                        'url': guide_url,
                    })
                    
                    logger.info(f"Found: {guide_number} - {title}")
            except Exception as e:
                logger.error(f"Error extracting guide card: {e}")
                continue
        
        logger.info(f"Found {len(guides)} guides")
        return guides
    
    def extract_table_of_contents(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract table of contents from guide page."""
        toc = []
        toc_nav = soup.find('div', class_='s_table_of_content_navbar')
        
        if not toc_nav:
            logger.warning("No table of contents found")
            return toc
        
        toc_links = toc_nav.find_all('a', class_='table_of_content_link')
        
        for link in toc_links:
            try:
                title = link.get_text(strip=True)
                anchor_id = link.get('href', '').lstrip('#')
                
                if title and anchor_id:
                    toc.append({
                        'title': title,
                        'anchor_id': anchor_id,
                    })
            except Exception as e:
                logger.error(f"Error extracting TOC item: {e}")
                continue
        
        return toc
    
    def extract_section_content(self, soup: BeautifulSoup, anchor_id: str) -> str:
        """Extract content for a specific section."""
        section = soup.find(id=anchor_id)
        
        if not section:
            return ""
        
        parent = section.find_parent()
        
        if not parent:
            return section.get_text(strip=True)
        
        content_parts = []
        
        for elem in parent.find_all(['p', 'ul', 'ol', 'li', 'div', 'span']):
            text = elem.get_text(strip=True)
            if text and len(text) > 10:
                content_parts.append(text)
        
        content = '\n\n'.join(content_parts)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        return content.strip()
    
    async def scrape_guide(self, guide_info: Dict[str, str]) -> Dict[str, Any]:
        """Scrape a single guide asynchronously.
        
        Args:
            guide_info: Dict with guide URL and metadata
            
        Returns:
            Dict with full guide data
        """
        url = guide_info['url']
        logger.info(f"Scraping guide: {guide_info['title']}")
        
        html = await self.fetch_page(url)
        
        if not html:
            return guide_info
        
        soup = self.parse_soup(html)
        
        # Extract table of contents
        toc = self.extract_table_of_contents(soup)
        
        # Extract content for each section
        sections = []
        for toc_item in toc:
            try:
                content = self.extract_section_content(soup, toc_item['anchor_id'])
                
                sections.append({
                    'title': toc_item['title'],
                    'anchor_id': toc_item['anchor_id'],
                    'content': content,
                    'content_length': len(content),
                })
            except Exception as e:
                logger.error(f"Error extracting section {toc_item['title']}: {e}")
                continue
        
        guide_data = {
            **guide_info,
            'table_of_contents': toc,
            'sections': sections,
            'total_sections': len(sections),
            'total_content_length': sum(s['content_length'] for s in sections),
        }
        
        logger.info(f"Scraped {len(sections)} sections from {guide_info['title']}")
        
        return guide_data
    
    async def scrape_all_guides(self) -> List[Dict[str, Any]]:
        """Scrape all guides in parallel.
        
        Returns:
            List of guide data dicts
        """
        await self.create_session()
        
        try:
            # Fetch main page
            logger.info(f"Fetching main page: {self.GUIDES_PAGE}")
            html = await self.fetch_page(self.GUIDES_PAGE)
            
            if not html:
                logger.error("Failed to fetch main page!")
                return []
            
            # Extract guide links
            guide_links = self.extract_guide_links(html)
            
            if not guide_links:
                logger.error("No guides found!")
                return []
            
            logger.info(f"Starting parallel scraping of {len(guide_links)} guides...")
            logger.info(f"Max concurrent requests: {self.max_concurrent}")
            logger.info(f"Delay between requests: {self.delay}s")
            
            # Scrape all guides in parallel with progress bar
            tasks = [self.scrape_guide(guide_info) for guide_info in guide_links]
            self.guides = await tqdm.gather(*tasks, desc="Scraping guides")
            
            logger.info(f"Successfully scraped {len(self.guides)} guides")
            
        finally:
            await self.close_session()
        
        return self.guides
    
    def create_searchable_text(self, guide: Dict[str, Any]) -> str:
        """Create searchable text from guide."""
        parts = []
        
        if guide.get('title'):
            parts.append(f"Guide: {guide['title']}")
        if guide.get('description'):
            parts.append(f"Description: {guide['description']}")
        
        for section in guide.get('sections', []):
            if section.get('title'):
                parts.append(f"\n## {section['title']}")
            if section.get('content'):
                parts.append(section['content'])
        
        return '\n\n'.join(parts)
    
    def save_guides(self, output_file: Path = None):
        """Save scraped guides to JSON file."""
        if not self.guides:
            logger.warning("No guides to save")
            return
        
        output_file = output_file or GUIDES_DATA_DIR / "guides.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        for guide in self.guides:
            guide['searchable_text'] = self.create_searchable_text(guide)
        
        logger.info(f"Saving {len(self.guides)} guides to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.guides, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved to {output_file}")
        
        return output_file
    
    def save_individual_guides(self, output_dir: Path = None):
        """Save each guide as a separate file."""
        if not self.guides:
            logger.warning("No guides to save")
            return
        
        output_dir = output_dir or GUIDES_DATA_DIR / "individual"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for guide in self.guides:
            try:
                guide_num = guide.get('guide_number', 'GUIDE_XX').replace(' ', '_')
                title_slug = re.sub(r'[^\w\s-]', '', guide.get('title', '')).strip()
                title_slug = re.sub(r'[-\s]+', '_', title_slug).lower()
                
                filename = f"{guide_num}_{title_slug}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(guide, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Saved {filepath.name}")
            except Exception as e:
                logger.error(f"Error saving guide {guide.get('title', 'unknown')}: {e}")
                continue
        
        logger.info(f"Saved {len(self.guides)} individual guides to {output_dir}")
        
        return output_dir
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about scraped guides."""
        if not self.guides:
            return {}
        
        total_sections = sum(g.get('total_sections', 0) for g in self.guides)
        total_content = sum(g.get('total_content_length', 0) for g in self.guides)
        
        stats = {
            'total_guides': len(self.guides),
            'total_sections': total_sections,
            'total_content_length': total_content,
            'avg_sections_per_guide': total_sections / len(self.guides) if self.guides else 0,
            'avg_content_per_guide': total_content / len(self.guides) if self.guides else 0,
            'guides': [
                {
                    'title': g.get('title', ''),
                    'sections': g.get('total_sections', 0),
                    'content_length': g.get('total_content_length', 0),
                }
                for g in self.guides
            ]
        }
        
        return stats


async def async_main():
    """Async main scraping function."""
    logger.info("Starting FAST Phase 3: Web Scraping (Parallel)")
    
    start_time = time.time()
    
    # Create scraper with 5 concurrent requests and 0.2s delay
    scraper = FastGuidesScraper(max_concurrent=5, delay=0.2)
    
    try:
        # Scrape all guides in parallel
        guides = await scraper.scrape_all_guides()
        
        if not guides:
            logger.error("No guides scraped!")
            return
        
        # Save guides
        json_file = scraper.save_guides()
        individual_dir = scraper.save_individual_guides()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print statistics
        stats = scraper.get_statistics()
        
        logger.info("\n" + "=" * 60)
        logger.info("Scraping Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Guides: {stats['total_guides']}")
        logger.info(f"Total Sections: {stats['total_sections']}")
        logger.info(f"Total Content Length: {stats['total_content_length']:,} characters")
        logger.info(f"Avg Sections per Guide: {stats['avg_sections_per_guide']:.1f}")
        logger.info(f"Processing Time: {elapsed_time:.2f} seconds")
        logger.info(f"Speed: {stats['total_guides'] / elapsed_time:.2f} guides/second")
        
        logger.info("\n" + "=" * 60)
        logger.info("Guides Scraped")
        logger.info("=" * 60)
        for guide in stats['guides']:
            logger.info(f"{guide['title']}: {guide['sections']} sections, {guide['content_length']:,} chars")
        
        logger.info("\n[SUCCESS] Fast scraping complete!")
        logger.info(f"   Combined JSON: {json_file}")
        logger.info(f"   Individual files: {individual_dir}")
        logger.info(f"   Total time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}", exc_info=True)
        raise


def main():
    """Main function with asyncio."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

