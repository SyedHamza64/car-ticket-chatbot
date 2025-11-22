"""Tests for Phase 3: Web Scraping."""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase3.scrape_guides import GuidesScraper


@pytest.fixture
def scraper():
    """Create GuidesScraper instance."""
    return GuidesScraper(delay=0)  # No delay for tests


@pytest.fixture
def sample_html_main_page():
    """Sample HTML for main guides page."""
    return """
    <html>
        <body>
            <div class="o_colored_level">
                <div class="card-title-12">
                    <b>GUIDE 01</b>
                </div>
                <h2 class="card-title-12">
                    <b>WASHING AND DRYING</b>
                </h2>
                <p class="card-text-1">
                    This is a test description.
                </p>
                <div class="btn-1">
                    <a href="/guide/lavaggio-ed-asciugatura" class="btn-primary-guide">
                        GO TO THE GUIDE
                    </a>
                </div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def sample_html_guide_page():
    """Sample HTML for a guide page (matches real structure)."""
    return """
    <html>
        <body>
            <div class="s_table_of_content_navbar">
                <a href="#table_of_content_heading_section1" class="table_of_content_link">Section 1</a>
                <a href="#table_of_content_heading_section2" class="table_of_content_link">Section 2</a>
            </div>
            <div class="container">
                <h2 id="table_of_content_heading_section1">Section 1</h2>
                <p>This is section 1 content with some details.</p>
                <p>More content for section 1 goes here.</p>
            </div>
            <div class="container">
                <h2 id="table_of_content_heading_section2">Section 2</h2>
                <p>This is section 2 content with more information.</p>
            </div>
        </body>
    </html>
    """


class TestGuidesScraper:
    """Test GuidesScraper class."""
    
    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.delay == 0
        assert scraper.BASE_URL == "https://www.lacuradellauto.it"
        assert scraper.guides == []
    
    def test_fetch_page_success(self, scraper, sample_html_main_page):
        """Test successful page fetch."""
        mock_response = Mock()
        mock_response.content = sample_html_main_page.encode('utf-8')
        mock_response.raise_for_status = Mock()
        
        scraper.session.get = Mock(return_value=mock_response)
        
        soup = scraper.fetch_page("https://test.com")
        
        assert soup is not None
        assert soup.find('b').get_text() == "GUIDE 01"
    
    def test_fetch_page_error(self, scraper):
        """Test page fetch error."""
        import requests
        scraper.session.get = Mock(side_effect=requests.RequestException("Network error"))
        
        soup = scraper.fetch_page("https://test.com")
        
        assert soup is None
    
    @patch.object(GuidesScraper, 'fetch_page')
    def test_extract_guide_links(self, mock_fetch, scraper, sample_html_main_page):
        """Test guide links extraction."""
        from bs4 import BeautifulSoup
        mock_fetch.return_value = BeautifulSoup(sample_html_main_page, 'html.parser')
        
        guides = scraper.extract_guide_links()
        
        assert len(guides) == 1
        assert guides[0]['guide_number'] == 'GUIDE 01'
        assert guides[0]['title'] == 'WASHING AND DRYING'
        assert 'lavaggio-ed-asciugatura' in guides[0]['url']
    
    def test_extract_table_of_contents(self, scraper, sample_html_guide_page):
        """Test TOC extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html_guide_page, 'html.parser')
        
        toc = scraper.extract_table_of_contents(soup)
        
        assert len(toc) == 2
        assert toc[0]['title'] == 'Section 1'
        assert toc[0]['anchor_id'] == 'table_of_content_heading_section1'
        assert toc[1]['title'] == 'Section 2'
        assert toc[1]['anchor_id'] == 'table_of_content_heading_section2'
    
    def test_extract_section_content(self, scraper, sample_html_guide_page):
        """Test section content extraction."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(sample_html_guide_page, 'html.parser')
        
        content = scraper.extract_section_content(soup, 'table_of_content_heading_section1')
        
        assert 'section 1 content' in content.lower()
        assert len(content) > 0
        # Should only get section 1 content, not section 2
        assert 'section 2' not in content.lower()
    
    def test_create_searchable_text(self, scraper):
        """Test searchable text creation."""
        guide = {
            'title': 'Test Guide',
            'description': 'Test description',
            'sections': [
                {
                    'title': 'Section 1',
                    'content': 'Section 1 content'
                },
                {
                    'title': 'Section 2',
                    'content': 'Section 2 content'
                }
            ]
        }
        
        searchable = scraper.create_searchable_text(guide)
        
        assert 'Test Guide' in searchable
        assert 'Test description' in searchable
        assert 'Section 1' in searchable
        assert 'Section 2' in searchable
    
    def test_get_statistics_empty(self, scraper):
        """Test statistics with no guides."""
        stats = scraper.get_statistics()
        
        assert stats == {}
    
    def test_get_statistics_with_guides(self, scraper):
        """Test statistics with guides."""
        scraper.guides = [
            {
                'title': 'Guide 1',
                'total_sections': 5,
                'total_content_length': 1000
            },
            {
                'title': 'Guide 2',
                'total_sections': 3,
                'total_content_length': 500
            }
        ]
        
        stats = scraper.get_statistics()
        
        assert stats['total_guides'] == 2
        assert stats['total_sections'] == 8
        assert stats['total_content_length'] == 1500
        assert stats['avg_sections_per_guide'] == 4.0

