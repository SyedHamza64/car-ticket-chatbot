"""
Full Diagnostic Suite - Runs all tests and generates optimization recommendations
"""
import json
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from diagnostics.latency_profiler import LatencyProfiler
from diagnostics.retrieval_quality_test import RetrievalQualityTest
from diagnostics.response_quality_test import ResponseQualityTest
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FullDiagnostic:
    """Run complete diagnostic suite and generate recommendations"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'latency': None,
            'retrieval': None,
            'response_quality': None,
            'recommendations': []
        }
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("\n" + "="*80)
        print("FULL DIAGNOSTIC SUITE - RAG System Analysis")
        print("="*80)
        print(f"\nStarting comprehensive diagnostics...")
        print("This will take several minutes...")
        
        # Test queries
        test_queries = [
            "Vorrei un aggiornamento per la spedizione del mio ordine del 9 settembre",
            "Con cleantle clean wheel acidic su cerchi, serve anche un wheel cleaner alcalino?",
            "La mia idropulitrice bigboi wash r ha il tubo che vibra molto forte",
            "Come posso lucidare la mia auto per rimuovere i graffi?",
            "Quali prodotti consigliate per pulire i sedili in pelle?",
            "I vetri della mia auto hanno macchie di calcare",
            "Come rimuovere lo sporco ferroso dai cerchi?",
        ]
        
        # 1. Latency Profiling
        print("\n" + "="*80)
        print("TEST 1/3: LATENCY PROFILING")
        print("="*80)
        profiler = LatencyProfiler()
        latency_results = profiler.run_test_suite(test_queries[:5], runs_per_query=2)
        self.results['latency'] = latency_results
        profiler.save_results(latency_results)
        
        # 2. Retrieval Quality
        print("\n" + "="*80)
        print("TEST 2/3: RETRIEVAL QUALITY")
        print("="*80)
        
        retrieval_test_cases = [
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
        
        retrieval_tester = RetrievalQualityTest()
        retrieval_results = retrieval_tester.run_test_suite(retrieval_test_cases)
        self.results['retrieval'] = retrieval_results
        retrieval_tester.save_results(retrieval_results)
        
        # 3. Response Quality
        print("\n" + "="*80)
        print("TEST 3/3: RESPONSE QUALITY")
        print("="*80)
        
        response_test_cases = [
            {
                'query': "Vorrei un aggiornamento per la spedizione del mio ordine del 9 settembre",
                'expected_elements': {
                    'required_keywords': ['ordine', 'spedizione'],
                    'forbidden_keywords': ['Nuovo Pelle', 'AutoClean'],
                    'min_length': 100,
                    'max_length': 800
                }
            },
            {
                'query': "Con cleantle clean wheel acidic su cerchi, serve anche un wheel cleaner alcalino?",
                'expected_elements': {
                    'required_keywords': ['cleantle', 'cerchi', 'acidic', 'alcalino'],
                    'forbidden_keywords': ['pulitacuoora'],
                    'min_length': 150,
                    'max_length': 800
                }
            },
            {
                'query': "Come posso lucidare la mia auto per rimuovere i graffi?",
                'expected_elements': {
                    'required_keywords': ['lucidare', 'lucidatura', 'graffi'],
                    'forbidden_keywords': ['Nuovo Pelle', 'AutoClean'],
                    'min_length': 200,
                    'max_length': 1000
                }
            },
            {
                'query': "Quali prodotti consigliate per pulire i sedili in pelle?",
                'expected_elements': {
                    'required_keywords': ['pelle', 'sedili', 'pulire'],
                    'forbidden_keywords': ['pulitacuoora', 'Nuovo Pelle'],
                    'min_length': 150,
                    'max_length': 800
                }
            },
            {
                'query': "Come rimuovere lo sporco ferroso dai cerchi?",
                'expected_elements': {
                    'required_keywords': ['ferroso', 'cerchi', 'iron'],
                    'forbidden_keywords': [],
                    'min_length': 150,
                    'max_length': 800
                }
            },
        ]
        
        response_tester = ResponseQualityTest()
        response_results = response_tester.run_test_suite(response_test_cases)
        self.results['response_quality'] = response_results
        response_tester.save_results(response_results)
        
        # 4. Generate recommendations
        print("\n" + "="*80)
        print("GENERATING OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        self.generate_recommendations()
        
        # Save full results
        self.save_full_results()
        
        # Print final recommendations
        self.print_recommendations()
    
    def generate_recommendations(self):
        """Generate prioritized optimization recommendations based on test results"""
        recommendations = []
        
        # Analyze latency results
        latency_data = self.results['latency']
        if latency_data:
            avg_total_ms = sum(r['average']['timings']['total_ms'] for r in latency_data) / len(latency_data)
            avg_llm_ms = sum(r['average']['timings']['llm_generation_ms'] for r in latency_data) / len(latency_data)
            llm_percentage = (avg_llm_ms / avg_total_ms) * 100
            
            if avg_total_ms > 10000:  # > 10 seconds
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Latency',
                    'issue': f'Very high latency: {avg_total_ms/1000:.1f}s average',
                    'recommendation': 'Stop generating 3 drafts in parallel - generate 1 draft only',
                    'expected_impact': '40-60% reduction in LLM time (8s → ~3.5-5s)',
                    'implementation': 'Set num_drafts=1 in Streamlit app',
                    'effort': 'IMMEDIATE (5 minutes)',
                    'reference': 'Quick Win #1'
                })
            
            if llm_percentage > 85:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Latency',
                    'issue': f'LLM takes {llm_percentage:.0f}% of total time',
                    'recommendation': 'Reduce num_predict from 500 to 200-250 and num_ctx to 1024',
                    'expected_impact': '20-40% faster generation',
                    'implementation': 'Update generate_response() in rag_pipeline.py',
                    'effort': 'IMMEDIATE (5 minutes)',
                    'reference': 'Quick Win #2'
                })
                
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Latency',
                    'issue': 'High token generation overhead',
                    'recommendation': 'Lower temperature to 0.3-0.5, top_k to 40',
                    'expected_impact': 'Faster sampling + more deterministic',
                    'implementation': 'Update generate_response() parameters',
                    'effort': 'IMMEDIATE (5 minutes)',
                    'reference': 'Quick Win #4'
                })
        
        # Analyze retrieval results
        retrieval_data = self.results['retrieval']
        if retrieval_data:
            avg_ticket_sim = [r['metrics'].get('avg_ticket_similarity', 0) for r in retrieval_data]
            avg_guide_sim = [r['metrics'].get('avg_guide_similarity', 0) for r in retrieval_data]
            avg_sim = (sum(avg_ticket_sim) + sum(avg_guide_sim)) / (len(avg_ticket_sim) + len(avg_guide_sim))
            
            topic_coverage = [r['metrics']['topic_coverage_pct'] for r in retrieval_data]
            avg_coverage = sum(topic_coverage) / len(topic_coverage)
            
            if avg_sim < 0.5:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Retrieval Quality',
                    'issue': f'Low embedding similarity: {avg_sim:.2f} average',
                    'recommendation': 'Switch to better embedding model: all-mpnet-base-v2',
                    'expected_impact': 'Improved retrieval accuracy, better context, 10-25% faster LLM',
                    'implementation': 'Update EMBEDDING_MODEL in settings.py and re-populate DB',
                    'effort': '1 WEEK',
                    'reference': 'Near-term #5'
                })
            
            if avg_coverage < 60:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Retrieval Quality',
                    'issue': f'Low topic coverage: {avg_coverage:.0f}%',
                    'recommendation': 'Add cross-encoder reranker to improve precision',
                    'expected_impact': 'Better context quality, fewer hallucinations',
                    'implementation': 'Integrate cross-encoder/ms-marco-MiniLM-L-6-v2',
                    'effort': '1 WEEK',
                    'reference': 'Near-term #7'
                })
        
        # Analyze response quality results
        response_data = self.results['response_quality']
        if response_data:
            avg_quality = sum(r['quality_score'] for r in response_data) / len(response_data)
            avg_keywords = sum(r['required_keywords_score'] for r in response_data) / len(response_data)
            avg_hallucination = sum(r['hallucination_score'] for r in response_data) / len(response_data)
            complete_count = sum(1 for r in response_data if r['complete'])
            
            if avg_quality < 70:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Response Quality',
                    'issue': f'Low overall quality score: {avg_quality:.0f}/100',
                    'recommendation': 'Improve prompt engineering with more examples and stricter instructions',
                    'expected_impact': 'Better adherence to guidelines, more accurate responses',
                    'implementation': 'Update create_prompt() with better examples',
                    'effort': 'IMMEDIATE (1 hour)',
                    'reference': 'Prompt engineering'
                })
            
            if avg_hallucination > 20:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Response Quality',
                    'issue': f'High hallucination rate: {avg_hallucination:.0f}%',
                    'recommendation': 'Strengthen anti-hallucination instructions in prompt',
                    'expected_impact': 'Fewer invented products, better trust',
                    'implementation': 'Add explicit product verification rules to prompt',
                    'effort': 'IMMEDIATE (30 minutes)',
                    'reference': 'Prompt engineering'
                })
            
            if avg_keywords < 70:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Response Quality',
                    'issue': f'LLM not using context well: {avg_keywords:.0f}% keyword coverage',
                    'recommendation': 'Compress retrieved docs to focus on key information',
                    'expected_impact': 'Better context utilization, faster generation',
                    'implementation': 'Add extractive summarization before prompt',
                    'effort': '1 WEEK',
                    'reference': 'Near-term #8'
                })
            
            if complete_count < len(response_data) * 0.8:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Response Quality',
                    'issue': f'Responses truncated: {complete_count}/{len(response_data)} complete',
                    'recommendation': 'This is already addressed by reducing num_predict to optimal range',
                    'expected_impact': 'Complete responses without wasted tokens',
                    'implementation': 'Set num_predict=250 (not 500)',
                    'effort': 'IMMEDIATE (5 minutes)',
                    'reference': 'Quick Win #2'
                })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        self.results['recommendations'] = recommendations
    
    def print_recommendations(self):
        """Print prioritized recommendations"""
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS (Prioritized)")
        print("="*80)
        
        immediate = [r for r in self.results['recommendations'] if r['effort'] == 'IMMEDIATE (5 minutes)' or r['effort'] == 'IMMEDIATE (30 minutes)' or r['effort'] == 'IMMEDIATE (1 hour)']
        short_term = [r for r in self.results['recommendations'] if r['effort'] == '1 WEEK']
        long_term = [r for r in self.results['recommendations'] if r['effort'] not in ['IMMEDIATE (5 minutes)', 'IMMEDIATE (30 minutes)', 'IMMEDIATE (1 hour)', '1 WEEK']]
        
        if immediate:
            print("\nIMMEDIATE ACTIONS (Do Now - Hours)")
            print("="*80)
            for i, rec in enumerate(immediate, 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Impact: {rec['expected_impact']}")
                print(f"   How: {rec['implementation']}")
                print(f"   Reference: {rec['reference']}")
        
        if short_term:
            print("\n\nSHORT-TERM (1-2 Weeks)")
            print("="*80)
            for i, rec in enumerate(short_term, 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Impact: {rec['expected_impact']}")
                print(f"   How: {rec['implementation']}")
                print(f"   Reference: {rec['reference']}")
        
        if long_term:
            print("\n\nLONG-TERM (1+ Months)")
            print("="*80)
            for i, rec in enumerate(long_term, 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Impact: {rec['expected_impact']}")
                print(f"   How: {rec['implementation']}")
        
        print("\n" + "="*80)
        print("RECOMMENDED PARAMETER CHANGES (Apply Immediately)")
        print("="*80)
        print("\nIn src/phase4/rag_pipeline.py - generate_response():")
        print("  num_predict: 500 -> 200-250")
        print("  num_ctx: 1536 -> 1024")
        print("  temperature: 0.7 -> 0.3-0.5")
        print("  top_k: 50 -> 40")
        print("\nIn streamlit_app.py:")
        print("  num_drafts: 3 -> 1 (default)")
        print("\n" + "="*80)
    
    def save_full_results(self):
        """Save complete diagnostic results"""
        output_path = Path("diagnostics/results/full_diagnostic.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nFull diagnostic results saved to: {output_path}")
        
        # Also save recommendations as separate file
        rec_path = Path("diagnostics/results/RECOMMENDATIONS.txt")
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("OPTIMIZATION RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            for rec in self.results['recommendations']:
                f.write(f"[{rec['priority']}] {rec['category']}\n")
                f.write(f"Issue: {rec['issue']}\n")
                f.write(f"Recommendation: {rec['recommendation']}\n")
                f.write(f"Expected Impact: {rec['expected_impact']}\n")
                f.write(f"Implementation: {rec['implementation']}\n")
                f.write(f"Effort: {rec['effort']}\n")
                f.write(f"Reference: {rec['reference']}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        logger.info(f"Recommendations saved to: {rec_path}")


if __name__ == "__main__":
    diagnostic = FullDiagnostic()
    diagnostic.run_all_tests()
    
    print("\n" + "="*80)
    print("FULL DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - diagnostics/results/full_diagnostic.json")
    print("  - diagnostics/results/latency_profile.json")
    print("  - diagnostics/results/retrieval_quality.json")
    print("  - diagnostics/results/response_quality.json")
    print("  - diagnostics/results/RECOMMENDATIONS.txt")
    print("\n" + "="*80)

