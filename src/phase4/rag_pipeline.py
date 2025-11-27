"""RAG Pipeline Orchestrator - Combines retrieval with LLM generation."""
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
# Removed ThreadPoolExecutor - using sequential generation for better GPU efficiency
import ollama
from src.phase4.vector_db import VectorDBManager
from src.utils.logger import setup_logger
from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = setup_logger(__name__)


class RAGPipeline:
    """RAG Pipeline for intelligent customer support responses."""
    
    def __init__(self, db_manager: Optional[VectorDBManager] = None, 
                 model: str = None, base_url: str = None):
        """Initialize RAG Pipeline.
        
        Args:
            db_manager: Vector database manager instance
            model: Ollama model name (default from config)
            base_url: Ollama base URL (default from config)
        """
        self.db_manager = db_manager or VectorDBManager()
        self.model = model or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        
        # Response cache (simple in-memory cache)
        self._cache = {}
        self._cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        
        logger.info(f"RAG Pipeline initialized with model: {self.model}")
        
        # Initialize collections if not already done
        if not self.db_manager.tickets_collection or not self.db_manager.guides_collection:
            self.db_manager.create_collections()
    
    def _get_cache_key(self, query: str, n_tickets: int, n_guides: int, language: str = "italian") -> str:
        """Generate cache key for a query."""
        cache_input = f"{query.lower().strip()}_{n_tickets}_{n_guides}_{language}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < self._cache_ttl:
                logger.info("Cache hit - returning cached response")
                return cached_data['response']
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache a response."""
        self._cache[cache_key] = {
            'response': response,
            'timestamp': datetime.now()
        }
        logger.info(f"Response cached (cache size: {len(self._cache)})")
    
    def retrieve_context(self, query: str, n_tickets: int = 3, n_guides: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context from vector database.
        
        Args:
            query: User query
            n_tickets: Number of relevant tickets to retrieve
            n_guides: Number of relevant guide sections to retrieve
            
        Returns:
            Retrieved context with tickets and guides
        """
        logger.info(f"Retrieving context for query: {query[:100]}...")
        
        results = self.db_manager.search_all(query, n_tickets=n_tickets, n_guides=n_guides)
        
        return results
    
    def format_context(self, results: Dict[str, Any]) -> str:
        """Format retrieved context for LLM prompt.
        
        Args:
            results: Search results from vector database
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add relevant tickets
        if results['tickets']['documents'] and results['tickets']['documents'][0]:
            context_parts.append("=== HISTORICAL TICKETS ===\n")
            for i, (doc, metadata) in enumerate(zip(
                results['tickets']['documents'][0],
                results['tickets']['metadatas'][0]
            ), 1):
                context_parts.append(f"\n[TICKET {i}]")
                context_parts.append(f"Subject: {metadata.get('subject', 'N/A')}")
                context_parts.append(f"Status: {metadata.get('status', 'N/A')}")
                context_parts.append(f"Content: {doc[:800]}...")  # Increased limit
                context_parts.append("")
        
        # Add relevant guides - show more content and highlight products
        if results['guides']['documents'] and results['guides']['documents'][0]:
            context_parts.append("\n=== TECHNICAL GUIDES ===\n")
            for i, (doc, metadata) in enumerate(zip(
                results['guides']['documents'][0],
                results['guides']['metadatas'][0]
            ), 1):
                doc_type = metadata.get('type', 'guide_section')
                guide_title = metadata.get('guide_title', 'N/A')
                section_title = metadata.get('section_title', 'N/A')
                
                context_parts.append(f"\n[GUIDE {i}]")
                context_parts.append(f"Guide: {guide_title}")
                
                # Show section type for clarity
                if doc_type == 'guide_tips':
                    context_parts.append(f"Tipo: Note e Suggerimenti Pratici")
                elif doc_type == 'guide_products':
                    context_parts.append(f"Tipo: Prodotti Consigliati")
                elif section_title:
                    context_parts.append(f"Sezione: {section_title}")
                
                # Show full content (increased from 700 to 1200)
                content = doc[:1200]
                if len(doc) > 1200:
                    content += "..."
                context_parts.append(f"Contenuto: {content}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str, language: str = "italian") -> str:
        """Create optimized prompt for LLM with context and query.
        
        Args:
            query: User query
            context: Retrieved and formatted context
            language: Response language ("italian" or "english")
            
        Returns:
            Complete prompt for LLM
        """
        if language.lower() == "english":
            prompt = f"""You are an expert AI assistant for LaCuraDellAuto customer support team. Your task is to help formulate professional and accurate responses to customer questions about car detailing products and techniques.

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== CUSTOMER QUESTION ===
{query}

=== INSTRUCTIONS ===
1. Carefully analyze the provided context (historical tickets and technical guides)
2. Formulate a clear, professional, and friendly response in English
3. **PRODUCTS**: When the question is about which product to use or how to solve a specific problem, recommend products from the context (e.g., "Gyeon Q2M Bathe", "Gtechniq C2"). DON'T force products if the question is purely informational
4. Cite techniques or steps from the guides when relevant
5. Use a friendly but professional tone, like an expert giving advice
6. Structure the response clearly (2-4 paragraphs, complete and detailed)
7. Make sure to complete all sentences and close the response professionally with a greeting
8. If the context doesn't contain sufficient information, say so clearly and suggest checking the catalog on the website
9. IMPORTANT: Use ONLY products mentioned in the context - do not invent product names
10. Recommend products when: the customer asks for advice, wants to solve a problem, or asks what to use. DON'T recommend products for purely theoretical questions

=== EXAMPLE OF GOOD RESPONSE ===
"Hello! To properly wash your car without scratching the paint, I recommend following these steps:

First of all, it's important to pre-wash the car with a water jet to remove surface dirt. This step is fundamental to avoid scratches during washing.

For the actual washing, use a specific car shampoo and a quality microfiber mitt. Work with linear movements rather than circular ones, and rinse the mitt frequently in the bucket.

Finally, dry the car with an absorbent microfiber cloth to avoid water spots. If you have other questions, I'm here to help!"

=== YOUR RESPONSE ==="""
        else:
            # Italian (default)
            prompt = f"""Sei un assistente AI esperto per il team di supporto clienti di LaCuraDellAuto. Il tuo compito è aiutare a formulare risposte professionali e accurate alle domande dei clienti riguardo prodotti e tecniche di car detailing.

=== CONTESTO DALLA BASE DI CONOSCENZA ===
{context}

=== DOMANDA DEL CLIENTE ===
{query}

=== ISTRUZIONI ===
1. Analizza attentamente il contesto fornito (ticket storici e guide tecniche)
2. Formula una risposta chiara, professionale e cordiale in italiano
3. **PRODOTTI**: Quando la domanda riguarda quale prodotto usare o come risolvere un problema specifico, raccomanda prodotti dal contesto (es: "Gyeon Q2M Bathe", "Gtechniq C2"). NON forzare prodotti se la domanda è solo informativa
4. Cita tecniche o passaggi dalle guide quando rilevante
5. Usa un tono amichevole ma professionale, come un esperto che consiglia
6. Struttura la risposta in modo chiaro (2-4 paragrafi, completa e dettagliata)
7. Assicurati di completare tutte le frasi e di chiudere la risposta in modo professionale con un saluto
8. Se il contesto non contiene informazioni sufficienti, dillo chiaramente e suggerisci di consultare il catalogo sul sito
9. IMPORTANTE: Usa SOLO prodotti menzionati nel contesto - non inventare nomi di prodotti
10. Raccomanda prodotti quando: il cliente chiede consigli, vuole risolvere un problema, o chiede cosa usare. NON raccomandare prodotti per domande puramente teoriche

=== ESEMPIO DI BUONA RISPOSTA ===
"Ciao! Per lavare correttamente la tua auto senza graffiare la vernice, ti consiglio di seguire questi passaggi:

Prima di tutto, è importante pre-lavare l'auto con un getto d'acqua per rimuovere lo sporco superficiale. Questo passaggio è fondamentale per evitare graffi durante il lavaggio.

Per il lavaggio vero e proprio, usa uno shampoo specifico per auto e un guanto in microfibra di qualità. Lavora con movimenti lineari piuttosto che circolari, e risciacqua frequentemente il guanto nel secchio.

Infine, asciuga l'auto con un panno in microfibra assorbente per evitare aloni. Se hai altre domande, sono qui per aiutarti!"

=== LA TUA RISPOSTA ==="""
        
        return prompt
    
    def generate_response(self, prompt: str, stream: bool = False, temperature: float = 0.7) -> str:
        """Generate response using Ollama LLM with optimized parameters.
        
        Args:
            prompt: Complete prompt with context
            stream: Whether to stream the response
            temperature: Creativity level (0.0-1.0). Higher = more creative/varied
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response with {self.model} (temp={temperature})")
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=stream,
                options={
                    'temperature': temperature,   # Configurable creativity
                    'top_p': 0.9,                 # Nucleus sampling
                    'top_k': 40,                  # Reduced for faster sampling (was 50)
                    'num_predict': 250,           # Optimized based on diagnostics (was 500)
                    'repeat_penalty': 1.1,        # Avoid repetition
                    'num_ctx': 1024,              # Optimized context window (was 1536)
                    'num_thread': 4,              # Use more CPU threads if GPU is busy
                }
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if 'response' in chunk:
                        full_response += chunk['response']
                return full_response
            else:
                return response['response']
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response. {str(e)}"
    
    def query(self, user_query: str, n_tickets: int = 3, n_guides: int = 3, 
              stream: bool = False, use_cache: bool = True, num_drafts: int = 1,
              language: str = "italian") -> Dict[str, Any]:
        """Main query method - retrieves context and generates response(s) with caching.
        
        Args:
            user_query: Customer query
            n_tickets: Number of relevant tickets to retrieve
            n_guides: Number of relevant guide sections to retrieve
            stream: Whether to stream the response
            use_cache: Whether to use cached responses
            num_drafts: Number of draft responses to generate (1-5)
            
        Returns:
            Dictionary with query, context, and generated response(s)
        """
        logger.info("="*80)
        logger.info(f"Processing query: {user_query}")
        logger.info("="*80)
        
        # Check cache first (only for non-streaming queries)
        if use_cache and not stream:
            cache_key = self._get_cache_key(user_query, n_tickets, n_guides, language)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Step 1: Retrieve relevant context
        results = self.retrieve_context(user_query, n_tickets, n_guides)
        
        # Step 2: Format context
        formatted_context = self.format_context(results)
        
        # Step 3: Create prompt
        prompt = self.create_prompt(user_query, formatted_context, language=language)
        
        # Step 4: Generate response(s)
        if num_drafts == 1:
            # Single response (original behavior)
            response = self.generate_response(prompt, stream, temperature=0.7)
            
            logger.info("Response generated successfully")
            
            result = {
                'query': user_query,
                'context': {
                    'tickets': results['tickets'],
                    'guides': results['guides']
                },
                'response': response,
                'model': self.model,
                'num_drafts': 1
            }
        else:
            # Multiple drafts with varying creativity - SEQUENTIAL GENERATION
            # Changed from parallel to sequential to avoid GPU context switching and memory thrash
            # This is actually faster for single GPU setups and reduces memory pressure
            logger.info(f"Generating {num_drafts} draft responses sequentially (optimized for single GPU)...")
            
            # Temperature variations for diversity
            # Draft 1: Conservative (0.3) - Most factual
            # Draft 2: Balanced (0.5) - Good middle ground
            # Draft 3: Slightly creative (0.7) - Some variety
            # Draft 4+: Additional variety
            temperatures = [0.3, 0.5, 0.7, 0.8, 0.9]
            
            # Generate drafts sequentially (one after another)
            # This avoids GPU context switching overhead and is actually faster on single GPU
            responses = []
            for i in range(num_drafts):
                draft_num = i + 1
                temperature = temperatures[min(i, len(temperatures) - 1)]
                
                try:
                    logger.info(f"Generating draft {draft_num}/{num_drafts} (temp={temperature})...")
                    text = self.generate_response(prompt, stream=False, temperature=temperature)
                    logger.info(f"Completed draft {draft_num}/{num_drafts}")
                    
                    responses.append({
                        'text': text,
                        'temperature': temperature,
                        'draft_number': draft_num
                    })
                except Exception as e:
                    logger.error(f"Draft {draft_num} generation failed: {e}")
                    responses.append({
                        'text': f"Error generating draft: {str(e)}",
                        'temperature': temperature,
                        'draft_number': draft_num
                    })
            
            logger.info(f"All {num_drafts} drafts generated successfully (sequential)")
            
            result = {
                'query': user_query,
                'context': {
                    'tickets': results['tickets'],
                    'guides': results['guides']
                },
                'response': responses[0]['text'],  # Primary response (for backwards compatibility)
                'responses': responses,  # All draft variations
                'model': self.model,
                'num_drafts': num_drafts
            }
        
        # Cache the result (only for non-streaming queries and single drafts)
        if use_cache and not stream and num_drafts == 1:
            self._cache_response(cache_key, result)
        
        return result
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available.
        
        Returns:
            Status dictionary
        """
        try:
            result = ollama.list()
            
            # Handle both old and new API format
            if isinstance(result, dict) and 'models' in result:
                models_list = result['models']
            else:
                models_list = result if isinstance(result, list) else []
            
            available_models = []
            for m in models_list:
                if isinstance(m, dict):
                    model_name = m.get('name', m.get('model', ''))
                    if model_name:
                        available_models.append(model_name)
            
            model_available = any(self.model in m for m in available_models)
            
            return {
                'ollama_running': True,
                'model_available': model_available,
                'available_models': available_models,
                'requested_model': self.model
            }
        except Exception as e:
            logger.error(f"Ollama status check failed: {e}")
            return {
                'ollama_running': False,
                'error': str(e)
            }

