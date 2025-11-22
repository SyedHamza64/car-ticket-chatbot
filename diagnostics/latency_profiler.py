"""
Latency Profiler - Measures detailed timing breakdown of RAG pipeline
"""
import time
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.phase4.rag_pipeline import RAGPipeline
from src.phase4.vector_db import VectorDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LatencyProfiler:
    """Profile latency of each RAG pipeline step"""
    
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.results = []
    
    def profile_query(self, query: str, n_tickets: int = 3, n_guides: int = 3) -> dict:
        """Profile a single query with detailed timing"""
        profile = {
            'query': query,
            'n_tickets': n_tickets,
            'n_guides': n_guides,
            'timings': {}
        }
        
        # Total time
        start_total = time.time()
        
        # 1. Cache check
        start = time.time()
        cache_key = self.pipeline._get_cache_key(query, n_tickets, n_guides)
        cached = self.pipeline._get_cached_response(cache_key)
        profile['timings']['cache_check_ms'] = (time.time() - start) * 1000
        profile['cache_hit'] = cached is not None
        
        if cached:
            profile['timings']['total_ms'] = (time.time() - start_total) * 1000
            return profile
        
        # 2. Query embedding generation
        start = time.time()
        query_embedding = self.pipeline.db_manager.generate_embeddings([query])
        profile['timings']['query_embedding_ms'] = (time.time() - start) * 1000
        
        # 3. Vector search (tickets)
        start = time.time()
        ticket_results = self.pipeline.db_manager.search_tickets(query, n_tickets)
        profile['timings']['search_tickets_ms'] = (time.time() - start) * 1000
        
        # 4. Vector search (guides)
        start = time.time()
        guide_results = self.pipeline.db_manager.search_guides(query, n_guides)
        profile['timings']['search_guides_ms'] = (time.time() - start) * 1000
        
        # 5. Context formatting
        start = time.time()
        results = {
            'tickets': ticket_results,
            'guides': guide_results
        }
        formatted_context = self.pipeline.format_context(results)
        profile['timings']['format_context_ms'] = (time.time() - start) * 1000
        
        # Context size metrics
        profile['context_stats'] = {
            'context_length_chars': len(formatted_context),
            'context_length_tokens_est': len(formatted_context.split()),
        }
        
        # 6. Prompt creation
        start = time.time()
        prompt = self.pipeline.create_prompt(query, formatted_context)
        profile['timings']['create_prompt_ms'] = (time.time() - start) * 1000
        
        # Prompt size metrics
        profile['prompt_stats'] = {
            'prompt_length_chars': len(prompt),
            'prompt_length_tokens_est': len(prompt.split()),
        }
        
        # 7. LLM generation
        start = time.time()
        response = self.pipeline.generate_response(prompt, temperature=0.7)
        profile['timings']['llm_generation_ms'] = (time.time() - start) * 1000
        
        # Response metrics
        profile['response_stats'] = {
            'response_length_chars': len(response),
            'response_length_tokens_est': len(response.split()),
        }
        
        # Total time
        profile['timings']['total_ms'] = (time.time() - start_total) * 1000
        
        # Calculate percentages
        total = profile['timings']['total_ms']
        profile['timings']['percentages'] = {
            'embedding': (profile['timings']['query_embedding_ms'] / total) * 100,
            'search': ((profile['timings']['search_tickets_ms'] + profile['timings']['search_guides_ms']) / total) * 100,
            'formatting': (profile['timings']['format_context_ms'] / total) * 100,
            'llm': (profile['timings']['llm_generation_ms'] / total) * 100,
        }
        
        return profile
    
    def run_test_suite(self, queries: list, runs_per_query: int = 3):
        """Run profiling on multiple queries"""
        all_results = []
        
        for query in queries:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {query[:60]}...")
            logger.info(f"{'='*80}")
            
            query_results = []
            for run in range(runs_per_query):
                logger.info(f"\nRun {run + 1}/{runs_per_query}")
                profile = self.profile_query(query)
                query_results.append(profile)
                
                # Print summary
                self._print_profile(profile)
            
            # Calculate averages for this query
            avg_profile = self._calculate_averages(query_results)
            all_results.append({
                'query': query,
                'runs': query_results,
                'average': avg_profile
            })
        
        return all_results
    
    def _calculate_averages(self, profiles: list) -> dict:
        """Calculate average timings across runs"""
        if not profiles:
            return {}
        
        avg = {
            'timings': {},
            'context_stats': {},
            'prompt_stats': {},
            'response_stats': {}
        }
        
        # Average timings
        timing_keys = profiles[0]['timings'].keys()
        for key in timing_keys:
            if key != 'percentages':
                values = [p['timings'][key] for p in profiles]
                avg['timings'][key] = sum(values) / len(values)
        
        # Average stats
        for stat_type in ['context_stats', 'prompt_stats', 'response_stats']:
            if stat_type in profiles[0]:
                for key in profiles[0][stat_type].keys():
                    values = [p[stat_type][key] for p in profiles if stat_type in p]
                    avg[stat_type][key] = sum(values) / len(values)
        
        return avg
    
    def _print_profile(self, profile: dict):
        """Print profile results"""
        if profile.get('cache_hit'):
            print(f"  Cache hit: {profile['timings']['total_ms']:.2f}ms")
            return
        
        print(f"\n  Timing Breakdown:")
        print(f"    Query Embedding:    {profile['timings']['query_embedding_ms']:>8.2f}ms ({profile['timings']['percentages']['embedding']:.1f}%)")
        print(f"    Search Tickets:     {profile['timings']['search_tickets_ms']:>8.2f}ms")
        print(f"    Search Guides:      {profile['timings']['search_guides_ms']:>8.2f}ms")
        print(f"    Total Search:       {profile['timings']['search_tickets_ms'] + profile['timings']['search_guides_ms']:>8.2f}ms ({profile['timings']['percentages']['search']:.1f}%)")
        print(f"    Format Context:     {profile['timings']['format_context_ms']:>8.2f}ms ({profile['timings']['percentages']['formatting']:.1f}%)")
        print(f"    Create Prompt:      {profile['timings']['create_prompt_ms']:>8.2f}ms")
        print(f"    LLM Generation:     {profile['timings']['llm_generation_ms']:>8.2f}ms ({profile['timings']['percentages']['llm']:.1f}%)")
        print(f"    ---------------------------------")
        print(f"    TOTAL:              {profile['timings']['total_ms']:>8.2f}ms ({profile['timings']['total_ms']/1000:.2f}s)")
        
        print(f"\n  Token/Size Estimates:")
        print(f"    Prompt tokens:      ~{profile['prompt_stats']['prompt_length_tokens_est']} tokens")
        print(f"    Response tokens:    ~{profile['response_stats']['response_length_tokens_est']} tokens")
    
    def save_results(self, results: list, output_file: str = "diagnostics/results/latency_profile.json"):
        """Save results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Test queries from real tickets
    test_queries = [
        "Buongiorno, vorrei chiedervi un aggiornamento per la spedizione del mio ordine",
        "Con cleantle clean wheel acidic su cerchi, è consigliabile anche una successiva erogazione di un qualche wheel cleaner alcalino?",
        "Ho acquistato una idropulitrice bigboi wash r e il tubo vibra molto forte",
        "Come posso lucidare la mia auto per rimuovere i graffi?",
        "Quali prodotti consigliate per pulire i sedili in pelle?",
    ]
    
    profiler = LatencyProfiler()
    
    print("\n" + "="*80)
    print("LATENCY PROFILER - RAG Pipeline Diagnostics")
    print("="*80)
    print(f"\nTesting {len(test_queries)} queries with 3 runs each...")
    
    results = profiler.run_test_suite(test_queries, runs_per_query=3)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Average Timings Across All Queries")
    print("="*80)
    
    all_totals = [r['average']['timings']['total_ms'] for r in results]
    all_llm = [r['average']['timings']['llm_generation_ms'] for r in results]
    all_search = [r['average']['timings']['search_tickets_ms'] + r['average']['timings']['search_guides_ms'] for r in results]
    all_embedding = [r['average']['timings']['query_embedding_ms'] for r in results]
    
    print(f"\nAverage Total Time:        {sum(all_totals)/len(all_totals):.2f}ms ({sum(all_totals)/len(all_totals)/1000:.2f}s)")
    print(f"Average LLM Time:          {sum(all_llm)/len(all_llm):.2f}ms ({(sum(all_llm)/sum(all_totals))*100:.1f}%)")
    print(f"Average Search Time:       {sum(all_search)/len(all_search):.2f}ms ({(sum(all_search)/sum(all_totals))*100:.1f}%)")
    print(f"Average Embedding Time:    {sum(all_embedding)/len(all_embedding):.2f}ms ({(sum(all_embedding)/sum(all_totals))*100:.1f}%)")
    
    # Save results
    profiler.save_results(results)

