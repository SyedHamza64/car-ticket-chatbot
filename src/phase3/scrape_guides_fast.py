"""Fast parallel scraper using async requests - Improved full-page extraction."""
import asyncio
import aiohttp
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin
import json

import cloudscraper
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm.asyncio import tqdm

from config.settings import GUIDES_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Boilerplate patterns to filter out
BOILERPLATE_PATTERNS = [
    r'function\s+\w+\s*\(',  # JavaScript functions
    r'var\s+\w+\s*=',  # JavaScript variables
    r'document\.getElementById',  # DOM manipulation
    r'window\.onload',  # Event handlers
    r'@media\s*\(',  # CSS media queries
    r'\.[\w-]+\s*\{',  # CSS rules
    r'padding-top:.*!important',  # CSS declarations
    r'Spedizioni Sicure.*Gratuite',  # Footer shipping text
    r'Affidabilità e Qualità',  # Footer quality text
    r'Supporto Rapido Dedicato',  # Footer support text
    r'In regalo la Guida al Detailing',  # Newsletter promo
    r'Accetto di ricevere comunicazioni',  # Newsletter checkbox
    r'Pagamenti e Spedizioni',  # Footer links
    r'Termini e Condizioni',  # Footer links
    r'Privacy e Cookie Policy',  # Footer links
    r'Preventivo Detailing',  # Footer links
    r'Centro Detailing a Bari',  # Footer links
    r'Via delle Mimose',  # Address
    r'P\.I\.\s*\d+',  # VAT number
    r'REA:\s*\d+',  # REA number
    r'PEC:.*@',  # PEC email
    r'Copyright\s*©',  # Copyright
    r'LaCuraDellAuto\.it',  # Brand in footer context
    r'__CF\$cv\$params',  # Cloudflare script
    r'cdn-cgi/challenge-platform',  # Cloudflare challenge
    r'tooltip\d+',  # Tooltip IDs
    r'aria-describedby',  # Accessibility attributes in text
    r'e ricevi uno sconto del \d+%!',  # Promo text
    r'ACQUISTA IL KIT',  # CTA buttons
    r'alert\(["\']',  # JavaScript alerts
]

BOILERPLATE_REGEX = re.compile('|'.join(BOILERPLATE_PATTERNS), re.IGNORECASE)


class FastGuidesScraper:
    """Fast parallel scraper for LaCuraDellAuto guides with improved extraction."""
    
    BASE_URL = "https://www.lacuradellauto.it"
    GUIDES_PAGE = f"{BASE_URL}/guida-detailing"
    
    # Content area selectors (main guide content)
    CONTENT_SELECTORS = [
        'div.col-lg-9.oe_structure',
        'div.s_table_of_content_main',
        'main',
        'article',
    ]
    
    # Elements to skip entirely
    SKIP_SELECTORS = [
        'footer',
        'nav',
        '.o_footer',
        '.s_newsletter',
        'script',
        'style',
        'noscript',
        'iframe',
        '.breadcrumb',
    ]
    
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
        self.scraper = cloudscraper.create_scraper(
            delay=10,
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
        
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
        """Fetch page content asynchronously."""
        async with self.semaphore:
            try:
                logger.debug(f"Fetching: {url}")
                
                async with self.session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    content = await response.text()
                    await asyncio.sleep(self.delay)
                    return content
            except Exception as e:
                logger.warning(f"Error fetching {url} via aiohttp: {e}. Falling back to cloudscraper...")
                return await self.fetch_with_cloudscraper(url)
    
    async def fetch_with_cloudscraper(self, url: str) -> Optional[str]:
        """Fallback fetch using cloudscraper to bypass protection layers."""
        def _fetch():
            try:
                response = self.scraper.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                logger.error(f"Cloudscraper error fetching {url}: {exc}")
                return None
        
        return await asyncio.to_thread(_fetch)
    
    def parse_soup(self, html: str) -> BeautifulSoup:
        """Parse HTML to BeautifulSoup."""
        return BeautifulSoup(html, 'html.parser')
    
    def extract_guide_links(self, html: str) -> List[Dict[str, str]]:
        """Extract all guide links from main page HTML."""
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
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing boilerplate and normalizing whitespace."""
        if not text:
            return ""
        
        # Check if text contains boilerplate patterns
        if BOILERPLATE_REGEX.search(text):
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Skip very short or very long single "words" (likely garbage)
        if len(text) < 10:
            return ""
        
        # Skip if mostly non-alphabetic
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return ""
        
        return text
    
    def _is_boilerplate_element(self, elem: Tag) -> bool:
        """Check if an element is boilerplate (footer, nav, etc.)."""
        if not isinstance(elem, Tag):
            return False
        
        # Check tag name
        if elem.name in ['script', 'style', 'noscript', 'iframe', 'nav', 'footer']:
            return True
        
        # Check classes
        classes = elem.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
        
        boilerplate_classes = [
            'o_footer', 's_newsletter', 'breadcrumb', 'cookie',
            'popup', 'modal', 'nav', 'menu', 'sidebar'
        ]
        
        for cls in classes:
            for bp in boilerplate_classes:
                if bp in cls.lower():
                    return True
        
        return False
    
    def _extract_products(self, elem: Tag) -> List[Dict[str, str]]:
        """Extract product links from an element."""
        products = []
        
        for link in elem.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Product links typically contain product IDs or .html
            if ('.html' in href or re.search(r'/\d+-', href)) and text:
                # Skip navigation/category links
                if any(skip in href.lower() for skip in ['category', 'guida', 'guide/', 'blog']):
                    continue
                
                full_url = urljoin(self.BASE_URL, href)
                products.append({
                    'name': text,
                    'url': full_url
                })
        
        # Deduplicate by URL
        seen_urls = set()
        unique_products = []
        for p in products:
            if p['url'] not in seen_urls:
                seen_urls.add(p['url'])
                unique_products.append(p)
        
        return unique_products
    
    def _extract_video_urls(self, soup: BeautifulSoup) -> List[str]:
        """Extract YouTube video URLs from the page."""
        videos = []
        
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if 'youtube.com' in src or 'youtu.be' in src:
                videos.append(src)
        
        # Also check for data attributes
        for elem in soup.find_all(attrs={'data-oe-expression': True}):
            expr = elem.get('data-oe-expression', '')
            if 'youtube.com' in expr:
                videos.append(expr)
        
        return list(set(videos))
    
    def _get_content_container(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content container."""
        for selector in self.CONTENT_SELECTORS:
            container = soup.select_one(selector)
            if container:
                return container
        
        # Fallback: find the largest text-containing div
        body = soup.find('body')
        if body:
            return body
        
        return None
    
    def _extract_section_from_element(self, section_elem: Tag, seen_text: Set[str], all_sections: List[Tag] = None, section_index: int = 0, skip_dedup_for_numbered: bool = True) -> Dict[str, Any]:
        """Extract content from a section element.
        
        Args:
            section_elem: The section element to extract from
            seen_text: Set of already seen text for deduplication
            all_sections: List of all sections (for sibling lookup)
            section_index: Index of this section in all_sections
            skip_dedup_for_numbered: Skip deduplication for numbered sections (01., 02., etc.)
        """
        heading = ""
        content_parts = []
        products = []
        
        # Find heading in this section
        for h_tag in ['h1', 'h2', 'h3', 'h4']:
            h_elem = section_elem.find(h_tag)
            if h_elem:
                heading = h_elem.get_text(strip=True)
                # Clean up multi-line headings
                heading = re.sub(r'\s+', ' ', heading).strip()
                break
        
        # Check if this is a numbered section (e.g., "01.", "02.", etc.)
        is_numbered_section = bool(re.match(r'^\d{2}\.', heading)) if heading else False
        
        # For numbered sections, use local deduplication only (don't check global seen_text)
        local_seen = set() if is_numbered_section and skip_dedup_for_numbered else None
        
        # Extract products from this section
        products = self._extract_products(section_elem)
        
        # Extract text content from this section
        for elem in section_elem.find_all(['p', 'li', 'div', 'span', 'blockquote']):
            # Skip if it's a container with children we'll process separately
            if elem.find(['p', 'li']) and elem.name == 'div':
                continue
            
            # Skip boilerplate
            if self._is_boilerplate_element(elem):
                continue
            
            text = elem.get_text(' ', strip=True)
            cleaned = self._clean_text(text)
            
            if cleaned and len(cleaned) > 20:
                normalized = re.sub(r'\s+', ' ', cleaned.lower())
                
                # Use local or global deduplication
                check_set = local_seen if local_seen is not None else seen_text
                
                is_duplicate = False
                for seen in check_set:
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break
                    if len(normalized) > 50 and len(seen) > 50:
                        common = len(set(normalized.split()) & set(seen.split()))
                        total = len(set(normalized.split()) | set(seen.split()))
                        if total > 0 and common / total > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    check_set.add(normalized)
                    if elem.name == 'li':
                        content_parts.append(f"• {cleaned}")
                    else:
                        content_parts.append(cleaned)
        
        # If this section has a heading but no content, check sibling sections
        if heading and not content_parts and all_sections:
            # Look at following sibling sections until we hit another heading
            for i in range(section_index + 1, len(all_sections)):
                sibling = all_sections[i]
                
                # Check if sibling has a heading (stop if it does)
                sibling_heading = sibling.find(['h1', 'h2', 'h3', 'h4'])
                if sibling_heading:
                    sibling_heading_text = sibling_heading.get_text(strip=True)
                    # Only stop if it's a real heading (not just the same heading repeated)
                    if sibling_heading_text and sibling_heading_text != heading:
                        break
                
                # Extract content from sibling
                sibling_products = self._extract_products(sibling)
                products.extend(sibling_products)
                
                for elem in sibling.find_all(['p', 'li', 'div', 'span', 'blockquote']):
                    if elem.find(['p', 'li']) and elem.name == 'div':
                        continue
                    if self._is_boilerplate_element(elem):
                continue
        
                    text = elem.get_text(' ', strip=True)
                    cleaned = self._clean_text(text)
                    
                    if cleaned and len(cleaned) > 20:
                        normalized = re.sub(r'\s+', ' ', cleaned.lower())
                        
                        # Use local deduplication for numbered sections
                        check_set = local_seen if local_seen is not None else seen_text
                        
                        is_duplicate = False
                        for seen in check_set:
                            if normalized in seen or seen in normalized:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            check_set.add(normalized)
                            if elem.name == 'li':
                                content_parts.append(f"• {cleaned}")
                            else:
                                content_parts.append(cleaned)
        
        content = '\n\n'.join(content_parts)
        
        return {
            'heading': heading,
            'content': content,
            'content_length': len(content),
            'products_mentioned': products
        }
    
    def _extract_content_after_heading(self, heading_elem: Tag, seen_text: Set[str]) -> str:
        """Extract content that follows a heading element (handles div containers)."""
        content_parts = []
        
        # Find the container div (parent or grandparent with 'container' class)
        container = heading_elem
        for _ in range(3):  # Check up to 3 levels up
            container = container.parent
            if not container:
                break
            classes = container.get('class', [])
            if isinstance(classes, str):
                classes = [classes]
            if 'container' in ' '.join(classes).lower():
                break
        
        if not container:
            return ""
        
        # Find all elements in the container
        all_elements = container.find_all(['p', 'ul', 'ol', 'section'], recursive=False)
        
        # Find the heading's position
        heading_found = False
        for elem in all_elements:
            # Check if this element contains or is the heading
            if heading_elem in elem.descendants or heading_elem == elem:
                heading_found = True
                continue
            
            if not heading_found:
                continue
            
            # Stop at next heading section
            if elem.name == 'section':
                next_heading = elem.find(['h1', 'h2', 'h3', 'h4'])
                if next_heading:
                    # Found next section with heading, stop here
                    break
            
            # Extract text from this element
            if elem.name == 'ul' or elem.name == 'ol':
                for li in elem.find_all('li', recursive=False):
                    text = self._clean_text(li.get_text(' ', strip=True))
                    if text and len(text) > 10:
                        normalized = re.sub(r'\s+', ' ', text.lower())
                        if normalized not in seen_text:
                            seen_text.add(normalized)
                            content_parts.append(f"• {text}")
            elif elem.name == 'p':
                text = self._clean_text(elem.get_text(' ', strip=True))
                if text and len(text) > 20:
                    normalized = re.sub(r'\s+', ' ', text.lower())
                    if normalized not in seen_text:
                        seen_text.add(normalized)
                        content_parts.append(text)
        
        return '\n\n'.join(content_parts)
    
    def extract_full_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract all content from the guide page using section-based approach."""
        result = {
            'intro': '',
            'sections': [],
            'tips': [],
            'video_urls': [],
            'all_products': []
        }
        
        # Get video URLs
        result['video_urls'] = self._extract_video_urls(soup)
        
        # Find main content container
        container = self._get_content_container(soup)
        if not container:
            logger.warning("Could not find content container")
            return result
        
        # Remove boilerplate elements
        for selector in self.SKIP_SELECTORS:
            for elem in container.select(selector):
                elem.decompose()
        
        # Track seen text for deduplication
        seen_text: Set[str] = set()
        all_products = []
        
        # Find all section elements AND div containers with content
        sections = container.find_all('section')
        content_divs = container.find_all('div', class_=lambda x: x and ('container' in ' '.join(x) if isinstance(x, list) else 'container' in x))
        
        # Combine and process all content blocks
        intro_parts = []
        current_sections = []
        tips = []
        
        # Process sections first
        for section in sections:
            # Skip empty sections
            if not section.get_text(strip=True):
                continue
            
            # Check for "Notes & Suggestions" section
            section_text = section.get_text(' ', strip=True).lower()
            if 'note' in section_text and 'suggeriment' in section_text:
                # Extract tips
                for li in section.find_all('li'):
                    tip_text = self._clean_text(li.get_text(' ', strip=True))
                    if tip_text and len(tip_text) > 20:
                        normalized = re.sub(r'\s+', ' ', tip_text.lower())
                        if normalized not in seen_text:
                            seen_text.add(normalized)
                            tips.append(tip_text)
                continue
            
            # Extract section content (pass all sections for sibling lookup)
            section_idx = sections.index(section) if section in sections else 0
            section_data = self._extract_section_from_element(section, seen_text, sections, section_idx)
            
            # If section has heading but no content, look for content after it
            if section_data['heading'] and not section_data['content']:
                heading_elem = section.find(['h1', 'h2', 'h3', 'h4'])
                if heading_elem:
                    # Find the container div that holds both sections and content
                    container = section.parent
                    for _ in range(3):
                        if not container:
                            break
                        classes = container.get('class', [])
                        if isinstance(classes, str):
                            classes = [classes]
                        if 'container' in ' '.join(classes).lower():
                            break
                        container = container.parent
                    
                    if container:
                        # Find all siblings after this section
                        all_siblings = container.find_all(['section', 'p', 'ul', 'ol', 'div'], recursive=False)
                        section_found = False
        content_parts = []
        
                        for sibling in all_siblings:
                            if sibling == section or section in sibling.descendants:
                                section_found = True
                                continue
                            
                            if not section_found:
                                continue
                            
                            # Stop at next heading section
                            if sibling.name == 'section':
                                next_heading = sibling.find(['h1', 'h2', 'h3', 'h4'])
                                if next_heading:
                                    break
                            
                            # Extract content from this sibling
                            if sibling.name == 'ul' or sibling.name == 'ol':
                                for li in sibling.find_all('li', recursive=False):
                                    text = self._clean_text(li.get_text(' ', strip=True))
            if text and len(text) > 10:
                                        normalized = re.sub(r'\s+', ' ', text.lower())
                                        if normalized not in seen_text:
                                            seen_text.add(normalized)
                                            content_parts.append(f"• {text}")
                            elif sibling.name == 'p':
                                text = self._clean_text(sibling.get_text(' ', strip=True))
                                if text and len(text) > 20:
                                    normalized = re.sub(r'\s+', ' ', text.lower())
                                    if normalized not in seen_text:
                                        seen_text.add(normalized)
                content_parts.append(text)
        
                        if content_parts:
                            section_data['content'] = '\n\n'.join(content_parts)
                            section_data['content_length'] = len(section_data['content'])
                            # Re-extract products from the content area
                            section_data['products_mentioned'] = self._extract_products(container)
            
            # Collect products
            all_products.extend(section_data['products_mentioned'])
            
            # Skip empty sections
            if not section_data['content'] and not section_data['heading']:
                continue
            
            # Check if this is intro content (before any real heading)
            if not section_data['heading'] and not current_sections:
                if section_data['content']:
                    intro_parts.append(section_data['content'])
            else:
                # Regular section
                if section_data['content'] or section_data['heading']:
                    current_sections.append(section_data)
        
        # Process div containers that might have content not in sections
        for div in content_divs:
            # Skip if already processed as part of a section
            if div.find_parent('section'):
                continue
            
            # Look for headings in this div
            headings = div.find_all(['h1', 'h2', 'h3', 'h4'])
            
            if not headings:
                # No headings, might be intro content
                text = self._clean_text(div.get_text(' ', strip=True))
                if text and len(text) > 50 and not current_sections:
                    normalized = re.sub(r'\s+', ' ', text.lower())
                    if normalized not in seen_text:
                        seen_text.add(normalized)
                        intro_parts.append(text)
                        all_products.extend(self._extract_products(div))
            else:
                # Process each heading and its following content
                for heading in headings:
                    heading_text = self._clean_text(heading.get_text(strip=True))
                    if not heading_text or len(heading_text) < 3:
                        continue
                    
                    # Extract content following this heading
                    content = self._extract_content_after_heading(heading, seen_text)
                    products = self._extract_products(div)
                    
                    if content or heading_text:
                        section_data = {
                            'heading': heading_text,
                            'content': content,
                            'content_length': len(content),
                            'products_mentioned': products
                        }
                        current_sections.append(section_data)
                        all_products.extend(products)
        
        result['intro'] = '\n\n'.join(intro_parts)
        result['sections'] = current_sections
        result['tips'] = tips
        
        # Deduplicate products
        seen_urls = set()
        unique_products = []
        for p in all_products:
            if p['url'] not in seen_urls:
                seen_urls.add(p['url'])
                unique_products.append(p)
        result['all_products'] = unique_products
        
        return result
    
    async def scrape_guide(self, guide_info: Dict[str, str]) -> Dict[str, Any]:
        """Scrape a single guide asynchronously with improved extraction."""
        url = guide_info['url']
        logger.info(f"Scraping guide: {guide_info['title']}")
        
        html = await self.fetch_page(url)
        
        if not html:
            return guide_info
        
        soup = self.parse_soup(html)
        
        # Extract all content using improved method
        content_data = self.extract_full_content(soup)
                
        # Build guide data
        guide_data = {
            **guide_info,
            'intro': content_data['intro'],
            'sections': content_data['sections'],
            'tips': content_data['tips'],
            'video_urls': content_data['video_urls'],
            'products_mentioned': content_data['all_products'],
            'total_sections': len(content_data['sections']),
            'total_content_length': (
                len(content_data['intro']) +
                sum(s['content_length'] for s in content_data['sections'])
            ),
        }
        
        logger.info(f"Scraped {len(content_data['sections'])} sections, "
                   f"{len(content_data['tips'])} tips, "
                   f"{len(content_data['all_products'])} products from {guide_info['title']}")
        
        return guide_data
    
    async def scrape_all_guides(self) -> List[Dict[str, Any]]:
        """Scrape all guides in parallel."""
        await self.create_session()
        
        try:
            logger.info(f"Fetching main page: {self.GUIDES_PAGE}")
            html = await self.fetch_page(self.GUIDES_PAGE)
            
            if not html:
                logger.error("Failed to fetch main page!")
                return []
            
            guide_links = self.extract_guide_links(html)
            
            if not guide_links:
                logger.error("No guides found!")
                return []
            
            logger.info(f"Starting parallel scraping of {len(guide_links)} guides...")
            logger.info(f"Max concurrent requests: {self.max_concurrent}")
            logger.info(f"Delay between requests: {self.delay}s")
            
            tasks = [self.scrape_guide(guide_info) for guide_info in guide_links]
            self.guides = await tqdm.gather(*tasks, desc="Scraping guides")
            
            logger.info(f"Successfully scraped {len(self.guides)} guides")
            
        finally:
            await self.close_session()
        
        return self.guides
    
    def create_searchable_text(self, guide: Dict[str, Any]) -> str:
        """Create clean searchable text from guide for vector embedding.
        
        Deduplicates content and creates a clean, structured text for LLM processing.
        """
        parts = []
        seen_content = set()  # Track seen content to avoid duplicates
        
        def add_unique_content(text: str, prefix: str = "") -> bool:
            """Add content only if not seen before (using first 100 chars as key)."""
            if not text or len(text.strip()) < 20:
                return False
            key = text.strip()[:100].lower()
            if key in seen_content:
                return False
            seen_content.add(key)
            parts.append(f"{prefix}{text}" if prefix else text)
            return True
        
        # Title and description
        if guide.get('title'):
            parts.append(f"Guida: {guide['title']}")
        if guide.get('description'):
            # Clean up description (remove extra whitespace)
            desc = ' '.join(guide['description'].split())
            add_unique_content(desc, "Descrizione: ")
        
        # Intro (skip if too similar to description)
        if guide.get('intro'):
            intro = ' '.join(guide['intro'].split())
            add_unique_content(intro, "\nIntroduzione:\n")
        
        # Sections - only include sections with meaningful content
        for section in guide.get('sections', []):
            heading = section.get('heading', '').strip()
            content = section.get('content', '').strip()
            
            # Skip empty sections or very short content
            if not content or len(content) < 50:
                continue
            
            # Skip if content is just the heading repeated
            if content.lower() == heading.lower():
                continue
            
            # Add heading if present and meaningful
            if heading and heading.lower() not in ['indice', 'index', '']:
                # Check if heading is not just a repetition
                if not any(heading.lower() in seen.lower() for seen in seen_content if len(seen) > 20):
                    parts.append(f"\n## {heading}")
            
            # Add content (deduplicated)
            add_unique_content(content)
        
        # Tips (practical advice - always valuable)
        if guide.get('tips'):
            parts.append("\n## Note e Suggerimenti")
            for tip in guide['tips']:
                tip_clean = tip.strip()
                if tip_clean and len(tip_clean) > 20:
                    # Check for duplicate tips
                    tip_key = tip_clean[:80].lower()
                    if tip_key not in seen_content:
                        seen_content.add(tip_key)
                        parts.append(f"• {tip_clean}")
        
        # Products mentioned (as keywords for retrieval)
        if guide.get('products_mentioned'):
            product_names = [p['name'] for p in guide['products_mentioned'] 
                          if p.get('name') and 'ACQUISTA' not in p.get('name', '').upper()]
            if product_names:
                parts.append(f"\nProdotti consigliati: {', '.join(product_names)}")
        
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
        total_tips = sum(len(g.get('tips', [])) for g in self.guides)
        total_products = sum(len(g.get('products_mentioned', [])) for g in self.guides)
        
        stats = {
            'total_guides': len(self.guides),
            'total_sections': total_sections,
            'total_content_length': total_content,
            'total_tips': total_tips,
            'total_products': total_products,
            'avg_sections_per_guide': total_sections / len(self.guides) if self.guides else 0,
            'avg_content_per_guide': total_content / len(self.guides) if self.guides else 0,
            'guides': [
                {
                    'title': g.get('title', ''),
                    'sections': g.get('total_sections', 0),
                    'tips': len(g.get('tips', [])),
                    'products': len(g.get('products_mentioned', [])),
                    'content_length': g.get('total_content_length', 0),
                }
                for g in self.guides
            ]
        }
        
        return stats


async def async_main():
    """Async main scraping function."""
    logger.info("Starting FAST Phase 3: Web Scraping (Improved Full-Page Extraction)")
    
    start_time = time.time()
    
    scraper = FastGuidesScraper(max_concurrent=5, delay=0.2)
    
    try:
        guides = await scraper.scrape_all_guides()
        
        if not guides:
            logger.error("No guides scraped!")
            return
        
        json_file = scraper.save_guides()
        individual_dir = scraper.save_individual_guides()
        
        elapsed_time = time.time() - start_time
        
        stats = scraper.get_statistics()
        
        logger.info("\n" + "=" * 60)
        logger.info("Scraping Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Guides: {stats['total_guides']}")
        logger.info(f"Total Sections: {stats['total_sections']}")
        logger.info(f"Total Tips: {stats['total_tips']}")
        logger.info(f"Total Products: {stats['total_products']}")
        logger.info(f"Total Content Length: {stats['total_content_length']:,} characters")
        logger.info(f"Avg Sections per Guide: {stats['avg_sections_per_guide']:.1f}")
        logger.info(f"Processing Time: {elapsed_time:.2f} seconds")
        logger.info(f"Speed: {stats['total_guides'] / elapsed_time:.2f} guides/second")
        
        logger.info("\n" + "=" * 60)
        logger.info("Guides Scraped")
        logger.info("=" * 60)
        for guide in stats['guides']:
            logger.info(f"{guide['title']}: {guide['sections']} sections, "
                       f"{guide['tips']} tips, {guide['products']} products, "
                       f"{guide['content_length']:,} chars")
        
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
