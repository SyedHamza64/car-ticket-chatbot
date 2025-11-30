#!/usr/bin/env python3
"""Rebuild Vector Database with tickets and guides."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase4.vector_db import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Rebuild the vector database from scratch."""
    print("=" * 60)
    print("REBUILDING VECTOR DATABASE")
    print("=" * 60)
    
    try:
        # Initialize vector DB manager
        print("\n[1/4] Initializing Vector Database Manager...")
        db_manager = VectorDBManager()
        
        # Reset and create fresh collections
        print("\n[2/4] Creating fresh collections (reset=True)...")
        db_manager.create_collections(reset=True)
        
        # Add tickets
        print("\n[3/4] Adding tickets to vector database...")
        try:
            db_manager.add_tickets()
            print("    ✓ Tickets added successfully")
        except FileNotFoundError as e:
            print(f"    ⚠ Tickets file not found: {e}")
        except Exception as e:
            print(f"    ✗ Error adding tickets: {e}")
        
        # Add guides
        print("\n[4/4] Adding guides to vector database...")
        try:
            db_manager.add_guides()
            print("    ✓ Guides added successfully")
        except FileNotFoundError as e:
            print(f"    ⚠ Guides file not found: {e}")
        except Exception as e:
            print(f"    ✗ Error adding guides: {e}")
        
        # Show stats
        print("\n" + "=" * 60)
        print("VECTOR DATABASE STATS")
        print("=" * 60)
        
        if db_manager.tickets_collection:
            ticket_count = db_manager.tickets_collection.count()
            print(f"  Tickets: {ticket_count}")
        
        if db_manager.guides_collection:
            guide_count = db_manager.guides_collection.count()
            print(f"  Guide chunks: {guide_count}")
        
        print("\n✅ Vector database rebuilt successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to rebuild vector database: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())



