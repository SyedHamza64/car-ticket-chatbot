"""
Process tickets only - Step 1: Clean and prepare tickets for embedding.
This script processes raw tickets and saves them to processed_tickets.json.
Does NOT generate embeddings or update ChromaDB.
"""
import json
import sys
import io
from pathlib import Path
import logging
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress duplicate logging from modules - disable their handlers
ticket_logger = logging.getLogger('src.phase2.process_tickets')
ticket_logger.setLevel(logging.WARNING)
ticket_logger.propagate = False  # Don't propagate to root logger
for handler in ticket_logger.handlers[:]:
    ticket_logger.removeHandler(handler)

# Configure logging to stdout for Streamlit visibility
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout,
    force=True  # Override any existing config
)

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
load_dotenv()

from config.settings import PROCESSED_DIR
from src.phase2.process_tickets import TicketProcessor

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("STEP 1: PROCESSING TICKETS")
    logger.info("=" * 60)
    logger.info("")
    
    # Load raw tickets - check environment variable first (for temp files from Streamlit)
    import os
    raw_file_path = os.environ.get('ZENDESK_EXPORT_FILE')
    
    if raw_file_path:
        raw_file = Path(raw_file_path)
    else:
        from config.settings import ZENDESK_EXPORT_FILE
        raw_file = ZENDESK_EXPORT_FILE
    
    if not raw_file.exists():
        logger.error(f"ERROR: Raw tickets file not found: {raw_file}")
        logger.info("Please upload tickets via Streamlit first.")
        return 1
    
    logger.info(f"Loading tickets from: {raw_file}")
    
    # Process tickets
    processor = TicketProcessor(input_file=raw_file)
    try:
        processor.load_tickets()
        logger.info(f"[OK] Loaded {len(processor.tickets)} raw tickets")
    except Exception as e:
        logger.error(f"ERROR: Error loading tickets: {e}")
        return 1
    
    logger.info("")
    logger.info("Processing tickets (cleaning, extracting conversation)...")
    
    try:
        processor.process_all()
        logger.info(f"[OK] Processed {len(processor.processed_tickets)} tickets")
    except Exception as e:
        logger.error(f"ERROR: Error processing tickets: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Merge with existing processed tickets
    processed_file = PROCESSED_DIR / "processed_tickets.json"
    existing_processed = []
    existing_processed_ids = set()
    
    if processed_file.exists():
        try:
            logger.info("")
            logger.info("Merging with existing processed tickets...")
            with open(processed_file, 'r', encoding='utf-8') as f:
                existing_processed = json.load(f)
                existing_processed_ids = {t.get('ticket_id') for t in existing_processed if t.get('ticket_id')}
            logger.info(f"[OK] Found {len(existing_processed)} existing processed tickets")
        except Exception as e:
            logger.warning(f"WARNING: Could not load existing processed tickets: {e}")
    
    # Add only new processed tickets
    new_count = 0
    for processed_ticket in processor.processed_tickets:
        ticket_id = processed_ticket.get('ticket_id')
        if ticket_id and ticket_id not in existing_processed_ids:
            existing_processed.append(processed_ticket)
            existing_processed_ids.add(ticket_id)
            new_count += 1
    
    # Save merged processed tickets
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("")
    logger.info(f"Saving {len(existing_processed)} total tickets ({new_count} new)...")
    
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(existing_processed, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[OK] Saved to: {processed_file}")
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 1 COMPLETE: {new_count} new tickets processed")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next: Run 'scripts/update_tickets_only.py' to generate embeddings and update ChromaDB")
    logger.info("")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

