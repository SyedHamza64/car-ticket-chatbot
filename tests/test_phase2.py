"""Tests for Phase 2: Zendesk Ticket Processing."""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.phase2.process_tickets import TicketProcessor


@pytest.fixture
def sample_ticket():
    """Sample ticket data for testing."""
    return {
        "id": 12345,
        "subject": "Test Ticket",
        "description": "<p>Test description</p>",
        "status": "closed",
        "priority": None,
        "via": {
            "channel": "email"
        },
        "created_at": "2025-01-01T10:00:00Z",
        "updated_at": "2025-01-01T11:00:00Z",
        "dates": {
            "solved_at": "2025-01-01T11:00:00Z"
        },
        "requester": {
            "id": 100,
            "name": "Test Customer",
            "email": "customer@test.com"
        },
        "assignee": {
            "id": 200,
            "name": "Test Agent",
            "email": "agent@test.com"
        },
        "comments": [
            {
                "id": 1,
                "type": "Comment",
                "author_id": 100,
                "body": "<p>Customer question</p>",
                "plain_body": "Customer question",
                "public": True,
                "created_at": "2025-01-01T10:00:00Z",
                "via": {
                    "channel": "email"
                }
            },
            {
                "id": 2,
                "type": "Comment",
                "author_id": 200,
                "body": "<p>Agent response</p>",
                "plain_body": "Agent response",
                "public": True,
                "created_at": "2025-01-01T10:30:00Z",
                "via": {
                    "channel": "email"
                }
            }
        ]
    }


@pytest.fixture
def processor():
    """Create TicketProcessor instance."""
    return TicketProcessor()


class TestTicketProcessor:
    """Test TicketProcessor class."""
    
    def test_clean_html(self, processor):
        """Test HTML cleaning."""
        html = "<p>Test <b>bold</b> text</p>"
        cleaned = processor.clean_html(html)
        assert cleaned == "Test bold text"
        assert "<" not in cleaned
        assert ">" not in cleaned
    
    def test_clean_html_empty(self, processor):
        """Test HTML cleaning with empty input."""
        assert processor.clean_html("") == ""
        assert processor.clean_html(None) == ""
    
    def test_extract_ticket_data(self, processor, sample_ticket):
        """Test ticket data extraction."""
        processor.tickets = [sample_ticket]
        processed = processor.extract_ticket_data(sample_ticket)
        
        assert processed['ticket_id'] == 12345
        assert processed['subject'] == "Test Ticket"
        assert processed['status'] == "closed"
        assert processed['channel'] == "email"
        assert processed['customer_name'] == "Test Customer"
        assert processed['agent_name'] == "Test Agent"
        assert processed['total_comments'] == 2
        assert len(processed['conversation']) == 2
        assert len(processed['customer_messages']) == 1
        assert len(processed['agent_messages']) == 1
    
    def test_create_searchable_text(self, processor, sample_ticket):
        """Test searchable text creation."""
        processor.tickets = [sample_ticket]
        processed = processor.extract_ticket_data(sample_ticket)
        searchable = processor.create_searchable_text(processed)
        
        assert "Test Ticket" in searchable
        assert "Customer question" in searchable
        assert "Agent response" in searchable
    
    def test_get_author_name_customer(self, processor, sample_ticket):
        """Test author name extraction for customer."""
        comment = sample_ticket['comments'][0]
        name = processor._get_author_name(comment, sample_ticket)
        assert name == "Test Customer"
    
    def test_get_author_name_agent(self, processor, sample_ticket):
        """Test author name extraction for agent."""
        comment = sample_ticket['comments'][1]
        name = processor._get_author_name(comment, sample_ticket)
        assert name == "Test Agent"
    
    def test_get_author_name_system(self, processor, sample_ticket):
        """Test author name extraction for system."""
        comment = {"author_id": -1}
        name = processor._get_author_name(comment, sample_ticket)
        assert name == "System"
    
    def test_process_all(self, processor, sample_ticket):
        """Test processing all tickets."""
        processor.tickets = [sample_ticket]
        processed = processor.process_all()
        
        assert len(processed) == 1
        assert processed[0]['ticket_id'] == 12345
        assert 'searchable_text' in processed[0]
    
    def test_get_statistics(self, processor, sample_ticket):
        """Test statistics generation."""
        processor.tickets = [sample_ticket]
        processor.process_all()
        stats = processor.get_statistics()
        
        assert stats['total_tickets'] == 1
        assert stats['channels']['email'] == 1
        assert stats['statuses']['closed'] == 1
        assert stats['total_comments'] == 2


class TestFileOperations:
    """Test file operations."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_load_tickets(self, mock_exists, mock_file, processor):
        """Test loading tickets from file."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps([{"id": 1}])
        
        tickets = processor.load_tickets()
        assert len(tickets) == 1
    
    @patch('pathlib.Path.exists')
    def test_load_tickets_file_not_found(self, mock_exists, processor):
        """Test loading tickets when file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            processor.load_tickets()

