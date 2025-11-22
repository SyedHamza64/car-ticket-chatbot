"""Populate vector database with tickets and guides."""
import sys
from src.phase4.vector_db import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main function to populate vector database."""
    logger.info("="*80)
    logger.info("Phase 4: Populating Vector Database")
    logger.info("="*80)
    
    try:
        # Initialize Vector DB Manager
        logger.info("\nStep 1: Initializing Vector Database...")
        db_manager = VectorDBManager()
        
        # Create collections (reset if needed)
        logger.info("\nStep 2: Creating Collections...")
        db_manager.create_collections(reset=True)
        
        # Add tickets
        logger.info("\nStep 3: Adding Tickets to Vector Database...")
        db_manager.add_tickets()
        
        # Add guides
        logger.info("\nStep 4: Adding Guides to Vector Database...")
        db_manager.add_guides()
        
        # Get stats
        logger.info("\n" + "="*80)
        logger.info("Vector Database Statistics")
        logger.info("="*80)
        stats = db_manager.get_stats()
        logger.info(f"Tickets in database: {stats['tickets']}")
        logger.info(f"Guide sections in database: {stats['guides']}")
        logger.info(f"Total documents: {sum(stats.values())}")
        
        logger.info("\n" + "="*80)
        logger.info("[SUCCESS] Vector Database Population Complete!")
        logger.info("="*80)
        logger.info("Next step: Test the RAG pipeline with queries")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n[ERROR] Failed to populate vector database: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

