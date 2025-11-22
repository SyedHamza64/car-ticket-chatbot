"""Process Zendesk tickets from JSON export."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup

from config.settings import ZENDESK_EXPORT_FILE, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TicketProcessor:
    """Process and clean Zendesk tickets."""
    
    def __init__(self, input_file: Path = None):
        """Initialize processor."""
        self.input_file = input_file or ZENDESK_EXPORT_FILE
        self.tickets = []
        self.processed_tickets = []
        
    def load_tickets(self) -> List[Dict[str, Any]]:
        """Load tickets from JSON file."""
        logger.info(f"Loading tickets from {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Zendesk export file not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)
        
        logger.info(f"Loaded {len(self.tickets)} tickets")
        return self.tickets
    
    def clean_html(self, html_text: str) -> str:
        """Remove HTML tags and clean text."""
        if not html_text:
            return ""
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_ticket_data(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure ticket data."""
        ticket_id = ticket.get('id')
        
        # Extract basic info
        basic_info = {
            'ticket_id': ticket_id,
            'subject': ticket.get('subject', ''),
            'description': self.clean_html(ticket.get('description', '')),
            'status': ticket.get('status', ''),
            'priority': ticket.get('priority'),
            'channel': ticket.get('via', {}).get('channel', ''),
            'created_at': ticket.get('created_at', ''),
            'updated_at': ticket.get('updated_at', ''),
            'solved_at': ticket.get('dates', {}).get('solved_at'),
        }
        
        # Extract customer info
        requester = ticket.get('requester', {})
        customer_info = {
            'customer_id': requester.get('id'),
            'customer_name': requester.get('name', ''),
            'customer_email': requester.get('email'),
        }
        
        # Extract agent info
        assignee = ticket.get('assignee', {})
        agent_info = {
            'agent_id': assignee.get('id'),
            'agent_name': assignee.get('name', ''),
            'agent_email': assignee.get('email'),
        }
        
        # Extract comments/conversation
        comments = ticket.get('comments', [])
        conversation = []
        
        for comment in comments:
            comment_data = {
                'comment_id': comment.get('id'),
                'type': comment.get('type', ''),
                'author_id': comment.get('author_id'),
                'author_name': self._get_author_name(comment, ticket),
                'body': self.clean_html(comment.get('body', '')),
                'plain_body': comment.get('plain_body', ''),
                'public': comment.get('public', False),
                'created_at': comment.get('created_at', ''),
                'via_channel': comment.get('via', {}).get('channel', ''),
            }
            conversation.append(comment_data)
        
        # Combine all data
        processed_ticket = {
            **basic_info,
            **customer_info,
            **agent_info,
            'conversation': conversation,
            'total_comments': len(conversation),
            'customer_messages': [c for c in conversation if c['author_id'] != assignee.get('id')],
            'agent_messages': [c for c in conversation if c['author_id'] == assignee.get('id')],
        }
        
        return processed_ticket
    
    def _get_author_name(self, comment: Dict, ticket: Dict) -> str:
        """Get author name from comment."""
        author_id = comment.get('author_id')
        
        # System/bot messages
        if author_id == -1:
            return "System"
        
        # Agent messages
        assignee = ticket.get('assignee', {})
        if author_id == assignee.get('id'):
            return assignee.get('name', 'Agent')
        
        # Customer messages
        requester = ticket.get('requester', {})
        if author_id == requester.get('id'):
            return requester.get('name', 'Customer')
        
        return "Unknown"
    
    def create_searchable_text(self, ticket: Dict[str, Any]) -> str:
        """Create searchable text from ticket for vector database."""
        parts = []
        
        # Add subject and description
        if ticket.get('subject'):
            parts.append(f"Subject: {ticket['subject']}")
        if ticket.get('description'):
            parts.append(f"Description: {ticket['description']}")
        
        # Add conversation
        for comment in ticket.get('conversation', []):
            author = comment.get('author_name', 'Unknown')
            body = comment.get('plain_body') or comment.get('body', '')
            if body:
                parts.append(f"{author}: {body}")
        
        return "\n\n".join(parts)
    
    def process_all(self) -> List[Dict[str, Any]]:
        """Process all tickets."""
        if not self.tickets:
            self.load_tickets()
        
        logger.info(f"Processing {len(self.tickets)} tickets...")
        
        self.processed_tickets = []
        for i, ticket in enumerate(self.tickets, 1):
            try:
                processed = self.extract_ticket_data(ticket)
                processed['searchable_text'] = self.create_searchable_text(processed)
                self.processed_tickets.append(processed)
                
                if i % 5 == 0:
                    logger.info(f"Processed {i}/{len(self.tickets)} tickets")
            except Exception as e:
                logger.error(f"Error processing ticket {ticket.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(self.processed_tickets)} tickets")
        return self.processed_tickets
    
    def save_to_json(self, output_file: Path = None):
        """Save processed tickets to JSON."""
        if not self.processed_tickets:
            self.process_all()
        
        output_file = output_file or PROCESSED_DATA_DIR / "processed_tickets.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.processed_tickets)} tickets to {output_file}")
        return output_file
    
    def save_to_csv(self, output_file: Path = None):
        """Save processed tickets to CSV (flattened)."""
        if not self.processed_tickets:
            self.process_all()
        
        output_file = output_file or PROCESSED_DATA_DIR / "processed_tickets.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten data for CSV
        flattened = []
        for ticket in self.processed_tickets:
            row = {
                'ticket_id': ticket['ticket_id'],
                'subject': ticket['subject'],
                'description': ticket['description'],
                'status': ticket['status'],
                'channel': ticket['channel'],
                'customer_name': ticket['customer_name'],
                'agent_name': ticket['agent_name'],
                'total_comments': ticket['total_comments'],
                'created_at': ticket['created_at'],
                'solved_at': ticket['solved_at'],
                'searchable_text': ticket['searchable_text'][:500],  # Truncate for CSV
            }
            flattened.append(row)
        
        df = pd.DataFrame(flattened)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(flattened)} tickets to {output_file}")
        return output_file
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed tickets."""
        if not self.processed_tickets:
            self.process_all()
        
        stats = {
            'total_tickets': len(self.processed_tickets),
            'channels': {},
            'statuses': {},
            'agents': {},
            'total_comments': 0,
            'total_customer_messages': 0,
            'total_agent_messages': 0,
        }
        
        for ticket in self.processed_tickets:
            # Channels
            channel = ticket.get('channel', 'unknown')
            stats['channels'][channel] = stats['channels'].get(channel, 0) + 1
            
            # Statuses
            status = ticket.get('status', 'unknown')
            stats['statuses'][status] = stats['statuses'].get(status, 0) + 1
            
            # Agents
            agent = ticket.get('agent_name', 'unknown')
            if agent:
                stats['agents'][agent] = stats['agents'].get(agent, 0) + 1
            
            # Comments
            stats['total_comments'] += ticket.get('total_comments', 0)
            stats['total_customer_messages'] += len(ticket.get('customer_messages', []))
            stats['total_agent_messages'] += len(ticket.get('agent_messages', []))
        
        return stats

def main():
    """Main processing function."""
    logger.info("Starting Phase 2: Zendesk Ticket Processing")
    
    processor = TicketProcessor()
    
    try:
        # Process tickets
        processor.process_all()
        
        # Save outputs
        json_file = processor.save_to_json()
        csv_file = processor.save_to_csv()
        
        # Print statistics
        stats = processor.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Processing Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Tickets: {stats['total_tickets']}")
        logger.info(f"Total Comments: {stats['total_comments']}")
        logger.info(f"Customer Messages: {stats['total_customer_messages']}")
        logger.info(f"Agent Messages: {stats['total_agent_messages']}")
        logger.info(f"\nChannels: {stats['channels']}")
        logger.info(f"Statuses: {stats['statuses']}")
        logger.info(f"Agents: {stats['agents']}")
        
        logger.info("\n[SUCCESS] Processing complete!")
        logger.info(f"   JSON: {json_file}")
        logger.info(f"   CSV: {csv_file}")
        
    except Exception as e:
        logger.error(f"Error processing tickets: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

