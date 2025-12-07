"""
Import new tickets from Zendesk export file.

This script:
1. Processes raw export file
2. Updates processed_tickets.json
3. Updates the knowledge base (ChromaDB + BM25)
"""
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
load_dotenv()

from config.settings import RAW_DIR, PROCESSED_DIR
from src.phase2.process_tickets import TicketProcessor
import subprocess

def merge_tickets(existing_file: Path, new_tickets: list):
    """Merge new tickets with existing, avoiding duplicates."""
    if not existing_file.exists():
        return new_tickets, len(new_tickets)
    
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)
    
    # Create a set of existing ticket IDs
    existing_ids = {t.get('ticket_id') for t in existing if t.get('ticket_id')}
    
    # Add new tickets that don't exist
    merged = existing.copy()
    added_count = 0
    
    for new_ticket in new_tickets:
        ticket_id = new_ticket.get('ticket_id')
        if ticket_id and ticket_id not in existing_ids:
            merged.append(new_ticket)
            existing_ids.add(ticket_id)
            added_count += 1
        elif ticket_id:
            # Update existing ticket (replace old with new)
            for i, old_ticket in enumerate(merged):
                if old_ticket.get('ticket_id') == ticket_id:
                    merged[i] = new_ticket
                    break
    
    return merged, added_count

def main():
    print("\n" + "=" * 60)
    print("📥 IMPORT NEW TICKETS FROM EXPORT")
    print("=" * 60 + "\n")
    
    # Step 1: Find export file
    export_file = RAW_DIR / "export_combined.json"
    
    if not export_file.exists():
        print(f"❌ Export file not found: {export_file}")
        print(f"\nPlease place your Zendesk export file at:")
        print(f"   {export_file}")
        print(f"\nOr specify a different path:")
        if len(sys.argv) > 1:
            export_file = Path(sys.argv[1])
        else:
            return 1
    
    print(f"📄 Found export file: {export_file}")
    
    # Step 2: Process export
    print("\n[1/4] Processing export file...")
    try:
        processor = TicketProcessor(input_file=export_file)
        processor.process_all()
        new_tickets = processor.processed_tickets
        print(f"✅ Processed {len(new_tickets)} tickets from export")
    except Exception as e:
        print(f"❌ Error processing export: {e}")
        return 1
    
    # Step 3: Merge with existing processed tickets
    print("\n[2/4] Merging with existing tickets...")
    processed_file = PROCESSED_DIR / "processed_tickets.json"
    
    try:
        merged_tickets, added_count = merge_tickets(processed_file, new_tickets)
        print(f"✅ Merged: {added_count} new tickets, {len(merged_tickets)} total")
    except Exception as e:
        print(f"❌ Error merging tickets: {e}")
        return 1
    
    # Step 4: Save processed tickets
    print("\n[3/4] Saving processed tickets...")
    try:
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(merged_tickets, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to: {processed_file}")
    except Exception as e:
        print(f"❌ Error saving processed tickets: {e}")
        return 1
    
    # Step 5: Update knowledge base
    print("\n[4/4] Updating knowledge base...")
    print("   (This may take 1-3 minutes)")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/update_tickets_only.py"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Knowledge base updated successfully!")
            # Print last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("\n   Last output:")
                for line in lines[-5:]:
                    print(f"   {line}")
        else:
            print(f"❌ Error updating knowledge base:")
            print(result.stderr)
            return 1
    except Exception as e:
        print(f"❌ Error running update script: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ IMPORT COMPLETE!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  • Processed: {len(new_tickets)} tickets from export")
    print(f"  • Added: {added_count} new tickets")
    print(f"  • Total: {len(merged_tickets)} tickets in KB")
    print(f"\nNext steps:")
    print(f"  1. Check stats in Streamlit")
    print(f"  2. Test queries to verify new tickets")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

