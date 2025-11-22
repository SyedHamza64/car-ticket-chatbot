#!/usr/bin/env python3
"""Quick benchmark script to compare different LLM models."""
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import ollama
from src.phase4.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Test query
TEST_QUERY = "Come posso lavare la mia auto senza graffiare la vernice?"

# Models to benchmark
MODELS_TO_TEST = [
    'gemma2:2b',                     # Current baseline
    'mixtral:8x7b-instruct-q4_K_M',  # Recommended
    'mistral:7b-instruct-q4_K_M',    # Alternative
]

def benchmark_model(model_name: str, num_runs: int = 3):
    """Benchmark a single model."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*80}")
    
    # Check if model is available
    try:
        models = ollama.list()
        available = False
        if isinstance(models, dict) and 'models' in models:
            model_list = models['models']
        else:
            model_list = models if isinstance(models, list) else []
        
        for m in model_list:
            if isinstance(m, dict):
                name = m.get('name', m.get('model', ''))
                if model_name in name or name == model_name:
                    available = True
                    print(f"Model found: {name}")
                    break
        
        if not available:
            print(f"WARNING: Model '{model_name}' not found in Ollama")
            print(f"Available models: {[m.get('name', m.get('model', '')) for m in model_list]}")
            print(f"Run: ollama pull {model_name}")
            return None
            
    except Exception as e:
        print(f"Error checking model availability: {e}")
        return None
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(model=model_name)
        print(f"Pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return None
    
    results = {
        'model': model_name,
        'single_query_times': [],
        'three_drafts_time': None,
        'errors': []
    }
    
    # Test 1: Single query (multiple runs)
    print(f"\n[Test 1] Single Query Latency ({num_runs} runs)...")
    for i in range(num_runs):
        try:
            start = time.time()
            result = pipeline.query(
                TEST_QUERY,
                n_tickets=3,
                n_guides=3,
                num_drafts=1,
                use_cache=False  # Disable cache for fair comparison
            )
            elapsed = time.time() - start
            
            results['single_query_times'].append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s (response length: {len(result['response'])} chars)")
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")
            results['errors'].append(str(e))
    
    # Test 2: Three drafts (parallel)
    print(f"\n[Test 2] Three Drafts (Parallel Generation)...")
    try:
        start = time.time()
        result = pipeline.query(
            TEST_QUERY,
            n_tickets=3,
            n_guides=3,
            num_drafts=3,
            use_cache=False
        )
        elapsed = time.time() - start
        results['three_drafts_time'] = elapsed
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Drafts: {len(result.get('responses', []))}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['errors'].append(str(e))
    
    # Calculate averages
    if results['single_query_times']:
        avg_time = sum(results['single_query_times']) / len(results['single_query_times'])
        min_time = min(results['single_query_times'])
        max_time = max(results['single_query_times'])
        
        results['avg_single_query'] = avg_time
        results['min_single_query'] = min_time
        results['max_single_query'] = max_time
        
        print(f"\n[Results]")
        print(f"  Single Query - Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")
        if results['three_drafts_time']:
            print(f"  Three Drafts: {results['three_drafts_time']:.2f}s")
            print(f"  Speedup (3x): {results['three_drafts_time'] / avg_time:.2f}x")
    
    return results


def main():
    """Run benchmarks for all models."""
    print("="*80)
    print("LLM Model Benchmarking")
    print("="*80)
    print(f"\nTest Query: {TEST_QUERY}")
    print(f"Models to test: {', '.join(MODELS_TO_TEST)}")
    
    all_results = []
    
    for model in MODELS_TO_TEST:
        result = benchmark_model(model)
        if result:
            all_results.append(result)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    if all_results:
        print(f"\n{'Model':<35} {'Avg Single':<12} {'3 Drafts':<12} {'Status'}")
        print("-"*80)
        
        baseline = None
        for r in all_results:
            status = "OK" if not r['errors'] else f"ERRORS: {len(r['errors'])}"
            
            avg = f"{r.get('avg_single_query', 0):.2f}s" if r.get('avg_single_query') else "N/A"
            three = f"{r['three_drafts_time']:.2f}s" if r.get('three_drafts_time') else "N/A"
            
            model_name = r['model'][:34]  # Truncate if too long
            print(f"{model_name:<35} {avg:<12} {three:<12} {status}")
            
            if 'gemma2:2b' in r['model']:
                baseline = r
        
        # Compare to baseline
        if baseline:
            print(f"\nComparison to baseline (gemma2:2b):")
            baseline_avg = baseline.get('avg_single_query', 0)
            baseline_three = baseline.get('three_drafts_time', 0)
            
            for r in all_results:
                if r['model'] != baseline['model'] and r.get('avg_single_query'):
                    speedup = baseline_avg / r['avg_single_query']
                    three_speedup = baseline_three / r['three_drafts_time'] if r.get('three_drafts_time') and baseline_three else None
                    
                    print(f"  {r['model']}:")
                    print(f"    Single query: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
                    if three_speedup:
                        print(f"    Three drafts: {three_speedup:.2f}x {'faster' if three_speedup > 1 else 'slower'}")
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)
    print("\nRecommendations:")
    print("  - If a model is not available, run: ollama pull <model-name>")
    print("  - Compare speed vs quality trade-offs")
    print("  - Test with your actual queries for best results")
    print()


if __name__ == "__main__":
    main()


