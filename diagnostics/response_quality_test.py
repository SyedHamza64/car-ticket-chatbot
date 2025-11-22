"""
Response Quality Test - Measures LLM response quality, hallucinations, and completeness
"""
import json
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent))

from src.phase4.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResponseQualityTest:
    """Test LLM response quality and accuracy"""
    
    def __init__(self):
        self.pipeline = RAGPipeline()
    
    def test_query(self, query: str, expected_elements: dict, true_answer: str = None) -> dict:
        """
        Test response quality for a query
        
        Args:
            query: User query
            expected_elements: Dict with:
                - required_keywords: Keywords that MUST be in response
                - forbidden_keywords: Keywords that should NOT be in response (hallucinations)
                - min_length: Minimum response length in chars
                - max_length: Maximum response length in chars
            true_answer: Optional true agent answer for comparison
        """
        result = {
            'query': query,
            'expected_elements': expected_elements,
            'true_answer': true_answer
        }
        
        # Generate response
        logger.info(f"Testing query: {query[:60]}...")
        response_data = self.pipeline.query(query, n_tickets=3, n_guides=3)
        
        response_text = response_data['response']
        result['response'] = response_text
        result['response_length'] = len(response_text)
        result['response_word_count'] = len(response_text.split())
        
        # Store retrieved context for analysis
        result['context'] = {
            'tickets_count': len(response_data['context']['tickets']['documents'][0]) if response_data['context']['tickets']['documents'] else 0,
            'guides_count': len(response_data['context']['guides']['documents'][0]) if response_data['context']['guides']['documents'] else 0
        }
        
        # Test 1: Required keywords present
        required = expected_elements.get('required_keywords', [])
        result['required_keywords_found'] = []
        result['required_keywords_missing'] = []
        
        for keyword in required:
            if keyword.lower() in response_text.lower():
                result['required_keywords_found'].append(keyword)
            else:
                result['required_keywords_missing'].append(keyword)
        
        result['required_keywords_score'] = (
            len(result['required_keywords_found']) / len(required) * 100
        ) if required else 100
        
        # Test 2: Forbidden keywords (potential hallucinations)
        forbidden = expected_elements.get('forbidden_keywords', [])
        result['forbidden_keywords_found'] = []
        
        for keyword in forbidden:
            if keyword.lower() in response_text.lower():
                result['forbidden_keywords_found'].append(keyword)
        
        result['hallucination_score'] = (
            len(result['forbidden_keywords_found']) / len(forbidden) * 100
        ) if forbidden else 0
        
        # Test 3: Length check
        min_len = expected_elements.get('min_length', 100)
        max_len = expected_elements.get('max_length', 2000)
        
        result['length_ok'] = min_len <= result['response_length'] <= max_len
        result['length_status'] = 'OK' if result['length_ok'] else ('TOO_SHORT' if result['response_length'] < min_len else 'TOO_LONG')
        
        # Test 4: Completeness (check if response ends properly)
        result['complete'] = self._check_completeness(response_text)
        
        # Test 5: Professional tone (has greeting and closing)
        result['has_greeting'] = bool(re.search(r'\b(ciao|buongiorno|salve)\b', response_text, re.IGNORECASE))
        result['has_closing'] = bool(re.search(r'\b(grazie|saluti|presto|aiutarti)\b', response_text, re.IGNORECASE))
        
        # Test 6: Product mentions (check if products are from context)
        result['product_mentions'] = self._extract_product_mentions(response_text)
        result['verified_products'] = self._verify_products_in_context(
            result['product_mentions'],
            response_data['context']
        )
        
        # Test 7: Structure (paragraphs)
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        result['paragraph_count'] = len(paragraphs)
        result['structured'] = 2 <= len(paragraphs) <= 5
        
        # Calculate overall quality score
        result['quality_score'] = self._calculate_quality_score(result)
        
        return result
    
    def _check_completeness(self, text: str) -> bool:
        """Check if response is complete (no mid-sentence cutoff)"""
        text = text.strip()
        if not text:
            return False
        
        # Check if ends with proper punctuation
        if text[-1] in '.!?':
            return True
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r'\.\.\.?$',  # Ends with ellipsis
            r'\w+$',      # Ends mid-word
            r',$',        # Ends with comma
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text):
                return False
        
        return True
    
    def _extract_product_mentions(self, text: str) -> list:
        """Extract product/brand names from text"""
        # Common patterns for product mentions
        products = []
        
        # Capitalized product names (simple heuristic)
        words = text.split()
        for i, word in enumerate(words):
            # Look for capitalized words that might be products
            if word[0].isupper() and len(word) > 3:
                # Check if it's not a sentence start
                if i > 0 and words[i-1][-1] not in '.!?':
                    products.append(word)
        
        return list(set(products))  # Remove duplicates
    
    def _verify_products_in_context(self, products: list, context: dict) -> dict:
        """Verify if mentioned products are in retrieved context"""
        verified = []
        unverified = []
        
        # Concatenate all context documents
        context_text = ""
        
        if context['tickets']['documents'] and context['tickets']['documents'][0]:
            context_text += " ".join(context['tickets']['documents'][0])
        
        if context['guides']['documents'] and context['guides']['documents'][0]:
            context_text += " ".join(context['guides']['documents'][0])
        
        context_text = context_text.lower()
        
        for product in products:
            if product.lower() in context_text:
                verified.append(product)
            else:
                unverified.append(product)
        
        return {
            'verified': verified,
            'unverified': unverified,
            'verification_rate': (len(verified) / len(products) * 100) if products else 100
        }
    
    def _calculate_quality_score(self, result: dict) -> float:
        """Calculate overall quality score (0-100)"""
        scores = []
        
        # Required keywords (weight: 30%)
        scores.append(('keywords', result['required_keywords_score'] * 0.3))
        
        # No hallucinations (weight: 25%)
        hallucination_penalty = result['hallucination_score']
        scores.append(('no_hallucination', (100 - hallucination_penalty) * 0.25))
        
        # Completeness (weight: 15%)
        scores.append(('completeness', 100 * 0.15 if result['complete'] else 0))
        
        # Length appropriate (weight: 10%)
        scores.append(('length', 100 * 0.10 if result['length_ok'] else 0))
        
        # Professional tone (weight: 10%)
        tone_score = 0
        if result['has_greeting']:
            tone_score += 50
        if result['has_closing']:
            tone_score += 50
        scores.append(('tone', tone_score * 0.10))
        
        # Structure (weight: 10%)
        scores.append(('structure', 100 * 0.10 if result['structured'] else 50 * 0.10))
        
        total_score = sum(s[1] for s in scores)
        
        return round(total_score, 2)
    
    def print_results(self, result: dict):
        """Print test results"""
        print(f"\n{'='*80}")
        print(f"Query: {result['query']}")
        print(f"{'='*80}")
        
        print(f"\nRESPONSE ({result['response_length']} chars, {result['response_word_count']} words):")
        print(f"  {result['response'][:300]}{'...' if len(result['response']) > 300 else ''}")
        
        print(f"\nREQUIRED KEYWORDS: {result['required_keywords_score']:.0f}%")
        if result['required_keywords_found']:
            print(f"  Found: {', '.join(result['required_keywords_found'])}")
        if result['required_keywords_missing']:
            print(f"  Missing: {', '.join(result['required_keywords_missing'])}")
        
        print(f"\nHALLUCINATION CHECK: {100 - result['hallucination_score']:.0f}% clean")
        if result['forbidden_keywords_found']:
            print(f"  Found forbidden: {', '.join(result['forbidden_keywords_found'])}")
        else:
            print(f"  No hallucinations detected")
        
        print(f"\nLENGTH: {result['length_status']} ({result['response_length']} chars)")
        print(f"COMPLETE: {'Yes' if result['complete'] else 'No (truncated)'}")
        print(f"PROFESSIONAL TONE: {('Yes' if result['has_greeting'] else 'No')} Greeting | {('Yes' if result['has_closing'] else 'No')} Closing")
        print(f"STRUCTURE: {result['paragraph_count']} paragraphs {'OK' if result['structured'] else 'NEEDS WORK'}")
        
        if result['product_mentions']:
            print(f"\nPRODUCT MENTIONS:")
            verified = result['verified_products']
            print(f"  Verified: {verified['verified'] if verified['verified'] else 'None'}")
            if verified['unverified']:
                print(f"  Unverified: {verified['unverified']} (potential hallucinations)")
        
        print(f"\nOVERALL QUALITY SCORE: {result['quality_score']:.1f}/100")
        
        # Quality rating
        if result['quality_score'] >= 80:
            print(f"   Rating: EXCELLENT")
        elif result['quality_score'] >= 60:
            print(f"   Rating: GOOD")
        elif result['quality_score'] >= 40:
            print(f"   Rating: FAIR")
        else:
            print(f"   Rating: POOR")
    
    def run_test_suite(self, test_cases: list):
        """Run multiple response quality tests"""
        results = []
        
        for test_case in test_cases:
            result = self.test_query(
                test_case['query'],
                test_case['expected_elements'],
                test_case.get('true_answer')
            )
            results.append(result)
            self.print_results(result)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: list):
        """Print summary of all tests"""
        print(f"\n{'='*80}")
        print("SUMMARY - Response Quality Across All Tests")
        print(f"{'='*80}")
        
        # Average scores
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        avg_keywords = sum(r['required_keywords_score'] for r in results) / len(results)
        avg_hallucination = sum(r['hallucination_score'] for r in results) / len(results)
        
        complete_count = sum(1 for r in results if r['complete'])
        length_ok_count = sum(1 for r in results if r['length_ok'])
        structured_count = sum(1 for r in results if r['structured'])
        
        print(f"\nAverage Quality Score:     {avg_quality:.1f}/100")
        print(f"Average Keyword Coverage:  {avg_keywords:.1f}%")
        print(f"Average Hallucination:     {avg_hallucination:.1f}%")
        print(f"Completeness:              {complete_count}/{len(results)} ({complete_count/len(results)*100:.0f}%)")
        print(f"Appropriate Length:        {length_ok_count}/{len(results)} ({length_ok_count/len(results)*100:.0f}%)")
        print(f"Well Structured:           {structured_count}/{len(results)} ({structured_count/len(results)*100:.0f}%)")
        
        print(f"\nRecommendations:")
        if avg_keywords < 70:
            print("  - LLM not using context well - improve prompt engineering")
        if avg_hallucination > 20:
            print("  - High hallucination rate - add stricter anti-hallucination instructions")
            print("  - Consider using retrieval-only mode or fact-checking layer")
        if complete_count < len(results) * 0.8:
            print("  - Increase num_predict to avoid truncation")
        if avg_quality < 70:
            print("  - Overall quality needs improvement - consider better base model")
    
    def save_results(self, results: list, output_file: str = "diagnostics/results/response_quality.json"):
        """Save results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            'query': "Vorrei un aggiornamento per la spedizione del mio ordine del 9 settembre",
            'expected_elements': {
                'required_keywords': ['ordine', 'spedizione'],
                'forbidden_keywords': ['Nuovo Pelle', 'AutoClean'],  # Hallucinated products
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
    
    tester = ResponseQualityTest()
    
    print("\n" + "="*80)
    print("RESPONSE QUALITY TEST - LLM Output Diagnostics")
    print("="*80)
    print(f"\nTesting {len(test_cases)} queries...")
    print("This may take a few minutes as each query requires LLM generation...")
    
    results = tester.run_test_suite(test_cases)
    
    # Save results
    tester.save_results(results)

