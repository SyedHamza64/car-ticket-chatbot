"""
Retrieval Quality Test - Measures how well the vector search retrieves relevant context
"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.phase4.vector_db import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RetrievalQualityTest:
    """Test retrieval quality of vector search"""
    
    def __init__(self):
        self.db_manager = VectorDBManager()
        self.db_manager.create_collections()
    
    def test_query(self, query: str, expected_topics: list, n_results: int = 3) -> dict:
        """
        Test retrieval for a query
        
        Args:
            query: User query
            expected_topics: List of keywords/topics that should be in retrieved docs
            n_results: Number of results to retrieve
        """
        result = {
            'query': query,
            'expected_topics': expected_topics,
            'n_results': n_results,
        }
        
        # Search tickets
        ticket_results = self.db_manager.search_tickets(query, n_results)
        
        # Search guides  
        guide_results = self.db_manager.search_guides(query, n_results)
        
        # Analyze ticket results
        result['tickets'] = {
            'count': len(ticket_results['ids'][0]) if ticket_results['ids'] else 0,
            'documents': [],
            'relevance_scores': []
        }
        
        if ticket_results['documents'] and ticket_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                ticket_results['documents'][0],
                ticket_results['metadatas'][0],
                ticket_results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                
                # Check if expected topics are in document
                topics_found = [topic for topic in expected_topics 
                               if topic.lower() in doc.lower()]
                
                result['tickets']['documents'].append({
                    'rank': i + 1,
                    'subject': metadata.get('subject', 'N/A'),
                    'similarity': similarity,
                    'distance': distance,
                    'topics_found': topics_found,
                    'snippet': doc[:200] + '...' if len(doc) > 200 else doc
                })
                
                result['tickets']['relevance_scores'].append(similarity)
        
        # Analyze guide results
        result['guides'] = {
            'count': len(guide_results['ids'][0]) if guide_results['ids'] else 0,
            'documents': [],
            'relevance_scores': []
        }
        
        if guide_results['documents'] and guide_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                guide_results['documents'][0],
                guide_results['metadatas'][0],
                guide_results['distances'][0]
            )):
                similarity = 1 - distance
                
                topics_found = [topic for topic in expected_topics 
                               if topic.lower() in doc.lower()]
                
                result['guides']['documents'].append({
                    'rank': i + 1,
                    'guide_title': metadata.get('guide_title', 'N/A'),
                    'section_title': metadata.get('section_title', 'N/A'),
                    'similarity': similarity,
                    'distance': distance,
                    'topics_found': topics_found,
                    'snippet': doc[:200] + '...' if len(doc) > 200 else doc
                })
                
                result['guides']['relevance_scores'].append(similarity)
        
        # Calculate metrics
        result['metrics'] = self._calculate_metrics(result)
        
        return result
    
    def _calculate_metrics(self, result: dict) -> dict:
        """Calculate retrieval quality metrics"""
        metrics = {}
        
        # Average similarity scores
        if result['tickets']['relevance_scores']:
            metrics['avg_ticket_similarity'] = sum(result['tickets']['relevance_scores']) / len(result['tickets']['relevance_scores'])
            metrics['min_ticket_similarity'] = min(result['tickets']['relevance_scores'])
            metrics['max_ticket_similarity'] = max(result['tickets']['relevance_scores'])
        
        if result['guides']['relevance_scores']:
            metrics['avg_guide_similarity'] = sum(result['guides']['relevance_scores']) / len(result['guides']['relevance_scores'])
            metrics['min_guide_similarity'] = min(result['guides']['relevance_scores'])
            metrics['max_guide_similarity'] = max(result['guides']['relevance_scores'])
        
        # Topic coverage
        total_docs = len(result['tickets']['documents']) + len(result['guides']['documents'])
        docs_with_topics = sum(
            1 for doc in result['tickets']['documents'] + result['guides']['documents']
            if doc['topics_found']
        )
        
        metrics['topic_coverage_pct'] = (docs_with_topics / total_docs * 100) if total_docs > 0 else 0
        
        # Best match
        all_docs = result['tickets']['documents'] + result['guides']['documents']
        if all_docs:
            best_match = max(all_docs, key=lambda x: x['similarity'])
            metrics['best_match'] = {
                'similarity': best_match['similarity'],
                'source': 'ticket' if best_match in result['tickets']['documents'] else 'guide',
                'topics_found': best_match['topics_found']
            }
        
        return metrics
    
    def print_results(self, result: dict):
        """Print retrieval test results"""
        print(f"\n{'='*80}")
        print(f"Query: {result['query']}")
        print(f"Expected Topics: {', '.join(result['expected_topics'])}")
        print(f"{'='*80}")
        
        # Tickets
        print(f"\nTICKETS (Top {result['n_results']}):")
        for doc in result['tickets']['documents']:
            print(f"  [{doc['rank']}] Similarity: {doc['similarity']:.3f} | Topics: {doc['topics_found'] or 'None'}")
            print(f"      Subject: {doc['subject']}")
            print(f"      Snippet: {doc['snippet'][:150]}...")
        
        # Guides
        print(f"\nGUIDES (Top {result['n_results']}):")
        for doc in result['guides']['documents']:
            print(f"  [{doc['rank']}] Similarity: {doc['similarity']:.3f} | Topics: {doc['topics_found'] or 'None'}")
            print(f"      Guide: {doc['guide_title']} > {doc['section_title']}")
            print(f"      Snippet: {doc['snippet'][:150]}...")
        
        # Metrics
        print(f"\nMETRICS:")
        metrics = result['metrics']
        if 'avg_ticket_similarity' in metrics:
            print(f"  Avg Ticket Similarity:  {metrics['avg_ticket_similarity']:.3f} (range: {metrics['min_ticket_similarity']:.3f} - {metrics['max_ticket_similarity']:.3f})")
        if 'avg_guide_similarity' in metrics:
            print(f"  Avg Guide Similarity:   {metrics['avg_guide_similarity']:.3f} (range: {metrics['min_guide_similarity']:.3f} - {metrics['max_guide_similarity']:.3f})")
        print(f"  Topic Coverage:         {metrics['topic_coverage_pct']:.1f}%")
        if 'best_match' in metrics:
            print(f"  Best Match:             {metrics['best_match']['similarity']:.3f} ({metrics['best_match']['source']}) - Topics: {metrics['best_match']['topics_found']}")
    
    def run_test_suite(self, test_cases: list):
        """Run multiple retrieval quality tests"""
        results = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_topics = test_case['expected_topics']
            
            result = self.test_query(query, expected_topics)
            results.append(result)
            self.print_results(result)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: list):
        """Print summary of all tests"""
        print(f"\n{'='*80}")
        print("SUMMARY - Retrieval Quality Across All Tests")
        print(f"{'='*80}")
        
        # Average metrics
        avg_ticket_sim = [r['metrics'].get('avg_ticket_similarity', 0) for r in results]
        avg_guide_sim = [r['metrics'].get('avg_guide_similarity', 0) for r in results]
        topic_coverage = [r['metrics']['topic_coverage_pct'] for r in results]
        
        print(f"\nAverage Ticket Similarity:  {sum(avg_ticket_sim)/len(avg_ticket_sim):.3f}")
        print(f"Average Guide Similarity:   {sum(avg_guide_sim)/len(avg_guide_sim):.3f}")
        print(f"Average Topic Coverage:     {sum(topic_coverage)/len(topic_coverage):.1f}%")
        
        # Quality assessment
        print(f"\nQuality Assessment:")
        
        avg_sim = (sum(avg_ticket_sim) + sum(avg_guide_sim)) / (len(avg_ticket_sim) + len(avg_guide_sim))
        avg_coverage = sum(topic_coverage) / len(topic_coverage)
        
        if avg_sim >= 0.7 and avg_coverage >= 70:
            print("  EXCELLENT - High similarity and good topic coverage")
        elif avg_sim >= 0.5 and avg_coverage >= 50:
            print("  GOOD - Decent similarity but room for improvement")
        elif avg_sim >= 0.3:
            print("  FAIR - Low similarity, consider better embeddings")
        else:
            print("  POOR - Very low similarity, retrieval needs improvement")
        
        print(f"\nRecommendations:")
        if avg_sim < 0.5:
            print("  - Consider using a better embedding model (all-mpnet-base-v2)")
            print("  - Increase n_results and add reranking")
        if avg_coverage < 60:
            print("  - Retrieved documents may not be relevant enough")
            print("  - Consider adding keyword boosting or hybrid search")
    
    def save_results(self, results: list, output_file: str = "diagnostics/results/retrieval_quality.json"):
        """Save results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Test cases with expected topics
    test_cases = [
        {
            'query': "Vorrei un aggiornamento per la spedizione del mio ordine",
            'expected_topics': ['ordine', 'spedizione', 'BRT', 'tracking']
        },
        {
            'query': "Con cleantle clean wheel acidic su cerchi, serve anche un wheel cleaner alcalino?",
            'expected_topics': ['cerchi', 'wheel', 'cleantle', 'alcalino', 'acidic', '2pH']
        },
        {
            'query': "La mia idropulitrice bigboi wash r ha il tubo che vibra molto forte",
            'expected_topics': ['idropulitrice', 'bigboi', 'tubo', 'vibra', 'filtro']
        },
        {
            'query': "Come posso lucidare la mia auto per rimuovere i graffi?",
            'expected_topics': ['lucidare', 'lucidatura', 'graffi', 'polish', 'vernice']
        },
        {
            'query': "Quali prodotti consigliate per pulire i sedili in pelle?",
            'expected_topics': ['pelle', 'sedili', 'interni', 'leather', 'pulire']
        },
        {
            'query': "I vetri della mia auto hanno macchie di calcare",
            'expected_topics': ['vetri', 'calcare', 'glass', 'macchie']
        },
        {
            'query': "Come rimuovere lo sporco ferroso dai cerchi?",
            'expected_topics': ['ferroso', 'cerchi', 'iron', 'remover', 'sporco']
        },
    ]
    
    tester = RetrievalQualityTest()
    
    print("\n" + "="*80)
    print("RETRIEVAL QUALITY TEST - Vector Search Diagnostics")
    print("="*80)
    print(f"\nTesting {len(test_cases)} queries...")
    
    results = tester.run_test_suite(test_cases)
    
    # Save results
    tester.save_results(results)

