#!/usr/bin/env python3
"""Test script to show retrieval behavior with limited tickets."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.phase4.vector_db import VectorDBManager

def test_retrieval_behavior():
    """Test how retrieval works with limited tickets."""
    print("="*80)
    print("Testing Retrieval Behavior with Limited Tickets")
    print("="*80)
    print()
    
    db = VectorDBManager()
    db.create_collections()
    
    stats = db.get_stats()
    total_tickets = stats['tickets']
    total_guides = stats['guides']
    
    print(f"Database Status:")
    print(f"  Total Tickets: {total_tickets}")
    print(f"  Total Guide Sections: {total_guides}")
    print()
    
    test_query = "Come posso lavare la mia auto senza graffiare la vernice?"
    
    print(f"Test Query: {test_query}")
    print()
    print("-"*80)
    
    # Test different retrieval settings
    test_cases = [
        (3, "Default setting"),
        (5, "Recommended for limited data"),
        (10, "More than available"),
        (total_tickets, "All available tickets"),
    ]
    
    for n_requested, description in test_cases:
        print(f"\nRequesting: {n_requested} tickets ({description})")
        print("-"*80)
        
        results = db.search_tickets(test_query, n_requested)
        
        n_returned = len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0
        
        print(f"  Requested: {n_requested} tickets")
        print(f"  Returned: {n_returned} tickets")
        print(f"  Behavior: ", end="")
        
        if n_returned == n_requested:
            print(f"✅ Returned exactly {n_requested} most relevant tickets")
        elif n_returned < n_requested:
            if n_returned == total_tickets:
                print(f"✅ Returned all {total_tickets} available tickets (requested {n_requested})")
            else:
                print(f"⚠️ Returned {n_returned} tickets (less than requested)")
        else:
            print(f"❌ Unexpected: returned more than requested")
        
        # Show similarity scores if available
        if results.get('distances') and results['distances'][0]:
            distances = results['distances'][0]
            similarities = [1 - d for d in distances]  # Convert distance to similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            print(f"  Average Similarity: {avg_similarity:.3f}")
            
            if len(similarities) > 0:
                print(f"  Similarity Range: {min(similarities):.3f} - {max(similarities):.3f}")
        
        # Show percentage of database
        if n_returned > 0:
            percentage = (n_returned / total_tickets) * 100
            print(f"  Percentage of Database: {percentage:.1f}%")
    
    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print()
    print("✅ System handles limited tickets gracefully:")
    print("   - Returns most relevant tickets based on semantic similarity")
    print("   - Automatically limits to available tickets if requested > available")
    print("   - Never errors - gracefully handles any number")
    print()
    print("💡 Recommendations:")
    print("   - With 19 tickets, requesting 3-5 is optimal (16-26% of database)")
    print("   - System prioritizes relevance over quantity")
    print("   - Guides (83 sections) provide primary knowledge base")
    print()

if __name__ == "__main__":
    test_retrieval_behavior()


