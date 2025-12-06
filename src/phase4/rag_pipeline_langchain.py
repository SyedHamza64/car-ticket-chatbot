"""
LangChain-powered RAG pipeline - Complete Migration
===================================================

Replaces custom RAG with LangChain primitives:
- LangChain HuggingFaceEmbeddings
- LangChain Chroma VectorStore
- LangChain-compatible BM25Retriever
- Hybrid retrieval with EnsembleRetriever
- CrossEncoder reranking
- LangChain LLM wrappers (Groq/Gemini/Ollama)
- Maintains compatibility with existing Chroma DB and BM25 index
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# LangChain 0.1.x compatible imports
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.schema import BaseRetriever
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.llms.base import LLM
    from langchain.prompts import PromptTemplate
except ImportError:
    # Try newer LangChain imports (0.2.x+)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.llms import LLM
        from langchain_core.prompts import PromptTemplate
    except ImportError as e:
        logger.error(f"Failed to import LangChain components: {e}")
        raise

try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    # Fallback for older LangChain versions
    EnsembleRetriever = None
    ContextualCompressionRetriever = None
    CrossEncoderReranker = None

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import requests

from config.settings import (
    CHROMA_DB_DIR,
    LOCAL_EMBEDDING_MODEL,
    BM25_INDEX_PATH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RERANKER_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM WRAPPERS (LangChain-compatible)
# ============================================================================

class GroqLLM(LLM):
    """LangChain LLM wrapper for Groq API."""
    
    model: str = GROQ_MODEL
    api_key: str = GROQ_API_KEY
    temperature: float = 0.2
    max_tokens: int = 1000
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful Italian car detailing support assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class GeminiLLM(LLM):
    """LangChain LLM wrapper for Google Gemini API."""
    
    model: str = GEMINI_MODEL
    api_key: str = GEMINI_API_KEY
    temperature: float = 0.4  # Slightly higher for more creative extraction
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Handle model name format
        if self.model.startswith("models/"):
            model_name = self.model
        else:
            model_name = f"models/{self.model}"
        
        url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2000,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            
            # Check for error response first
            if "error" in result:
                error_info = result.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                error_code = error_info.get("code", "UNKNOWN")
                logger.error(f"Gemini API returned error: {error_code} - {error_msg}")
                raise ValueError(f"Gemini API error ({error_code}): {error_msg}")
            
            # Log response for debugging (only structure, not full content)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Gemini API response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Handle different response formats
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                # Check for finishReason (might be blocked)
                if "finishReason" in candidate:
                    finish_reason = candidate["finishReason"]
                    if finish_reason != "STOP":
                        logger.warning(f"Gemini finish reason: {finish_reason}")
                        if finish_reason == "SAFETY":
                            raise ValueError("Gemini blocked the response due to safety settings")
                        elif finish_reason == "MAX_TOKENS":
                            logger.warning("Gemini hit max tokens limit")
                        elif finish_reason == "RECITATION":
                            raise ValueError("Gemini blocked due to recitation policy")
                
                # Extract text from content
                if "content" in candidate:
                    content = candidate["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        text = content["parts"][0].get("text", "")
                        if text:
                            return text
                
                # Try alternative path
                if "text" in candidate:
                    return candidate["text"]
            
            # If we get here, log the full response for debugging
            logger.error(f"Unexpected Gemini API response format.")
            logger.error(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            logger.error(f"Full response (first 500 chars): {str(result)[:500]}")
            
            # Try to provide helpful error message
            if "error" in result:
                error_info = result.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                raise ValueError(f"Gemini API error: {error_msg}")
            
            raise ValueError(f"Unexpected Gemini API response format. Response structure: {type(result)}. Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Gemini API HTTP error: {e}"
            try:
                error_detail = e.response.json()
                error_msg += f" Details: {error_detail}"
            except:
                error_msg += f" Response: {e.response.text}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class OllamaLLM(LLM):
    """LangChain LLM wrapper for local Ollama."""
    
    model: str = OLLAMA_MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.2
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


# ============================================================================
# BM25 RETRIEVER (LangChain-compatible)
# ============================================================================

class BM25Retriever(BaseRetriever):
    """LangChain-compatible BM25 retriever using existing pickle index."""
    
    # Store as instance attributes (not Pydantic fields)
    def __init__(self, bm25_obj: BM25Okapi, ids: List[str], docs: List[str], k: int = 50):
        # Don't call super().__init__() with Pydantic validation
        # Set attributes directly
        object.__setattr__(self, 'bm25', bm25_obj)
        object.__setattr__(self, 'ids', ids)
        object.__setattr__(self, 'docs', docs)
        object.__setattr__(self, 'k', k)
        # Add attributes required by BaseRetriever
        object.__setattr__(self, 'tags', [])
        object.__setattr__(self, 'metadata', {})
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Retrieve relevant documents using BM25."""
        if not self.bm25 or not self.ids:
            return []
        
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.k]
        
        documents = []
        for idx, score in ranked_indices:
            if idx < len(self.docs) and idx < len(self.ids):
                doc_text = self.docs[idx]
                doc_id = self.ids[idx]
                
                # Extract metadata from ID if possible
                metadata = {
                    "bm25_score": float(score),
                    "source_id": doc_id,
                }
                
                # Try to parse type from ID format
                if "_ticket_" in doc_id:
                    metadata["type"] = "ticket"
                    # Extract ticket_id if possible
                    parts = doc_id.split("_ticket_")
                    if len(parts) > 1:
                        metadata["orig_ticket_id"] = parts[-1]
                elif "_guide_" in doc_id:
                    metadata["type"] = "guide_chunk"
                
                documents.append(Document(page_content=doc_text, metadata=metadata))
        
        return documents


# ============================================================================
# CROSS-ENCODER RERANKER
# ============================================================================

class CrossEncoderRerankerWrapper:
    """Wrapper for CrossEncoder reranking compatible with LangChain."""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        try:
            self.ce = CrossEncoder(model_name)
            logger.info(f"Loaded CrossEncoder reranker: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}")
            self.ce = None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Rerank documents using CrossEncoder."""
        if not self.ce or not documents:
            return documents[:top_k]
        
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.ce.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]


# ============================================================================
# MAIN LANGCHAIN RAG PIPELINE
# ============================================================================

class LangchainRAG:
    """
    Complete LangChain-powered RAG pipeline.
    
    Features:
    - LangChain HuggingFaceEmbeddings
    - LangChain Chroma VectorStore
    - BM25 sparse retrieval
    - Hybrid dense + sparse retrieval
    - CrossEncoder reranking
    - Support for Groq/Gemini/Ollama LLMs
    - Separate ticket/guide retrieval
    """
    
    def __init__(
        self,
        provider: str = "grok",
        model: str = None,
        chroma_dir: Optional[Path] = None,
        hf_model_name: Optional[str] = None,
        bm25_path: Optional[Path] = None,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        sparse_weight: float = HYBRID_SPARSE_WEIGHT,
        reranker_model: Optional[str] = RERANKER_MODEL,
    ):
        """
        Initialize LangChain RAG pipeline.
        
        Args:
            provider: LLM provider - "grok", "gemini", or "ollama"
            model: Model name (optional, uses default from config)
            chroma_dir: Path to Chroma DB directory
            hf_model_name: HuggingFace embedding model name
            bm25_path: Path to BM25 pickle file
            dense_weight: Weight for dense retrieval in hybrid search
            sparse_weight: Weight for sparse retrieval in hybrid search
            reranker_model: CrossEncoder model for reranking
        """
        self.provider = provider.lower()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Set model
        if model:
            self.model_name = model
        elif self.provider == "grok":
            self.model_name = GROQ_MODEL
        elif self.provider == "gemini":
            self.model_name = GEMINI_MODEL
        elif self.provider == "ollama":
            self.model_name = OLLAMA_MODEL
        else:
            self.model_name = GROQ_MODEL
        
        # 1. Initialize embeddings
        hf_model = hf_model_name or LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading HuggingFace embeddings: {hf_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. Initialize Chroma vectorstore
        chroma_path = chroma_dir or CHROMA_DB_DIR
        logger.info(f"Loading Chroma DB from: {chroma_path}")
        
        # Try unified collection first, then fallback to separate collections
        try:
            self.chroma = Chroma(
                collection_name="rag_v2",
                embedding_function=self.embeddings,
                persist_directory=str(chroma_path),
            )
            logger.info("Using unified 'rag_v2' collection")
        except Exception as e:
            logger.warning(f"Could not load 'rag_v2' collection: {e}. Trying separate collections.")
            self.chroma = Chroma(
                collection_name="tickets",
                embedding_function=self.embeddings,
                persist_directory=str(chroma_path),
            )
        
        # Create retrievers for tickets and guides separately
        # Increase k to get more candidates for better retrieval
        self.tickets_retriever = self.chroma.as_retriever(
            search_kwargs={"k": 300, "filter": {"type": "ticket"}}
        )
        self.guides_retriever = self.chroma.as_retriever(
            search_kwargs={"k": 300, "filter": {"type": "guide_chunk"}}
        )
        
        # Fallback: if filter doesn't work, use general retriever
        try:
            _ = self.tickets_retriever.get_relevant_documents("test")
        except Exception:
            logger.warning("Chroma filter not supported, using general retriever")
            self.general_retriever = self.chroma.as_retriever(search_kwargs={"k": 400})
            self.tickets_retriever = None
            self.guides_retriever = None
        
        # 3. Load BM25 index
        bm25_file = bm25_path or BM25_INDEX_PATH
        self.bm25_retriever = None
        
        if bm25_file.exists():
            logger.info(f"Loading BM25 index from: {bm25_file}")
            with open(bm25_file, "rb") as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                bm25_obj = data.get("bm25")
                bm25_ids = data.get("ids", [])
                bm25_docs = data.get("docs", [])
            else:
                bm25_obj, bm25_ids = data
                bm25_docs = None
            
            if bm25_obj and bm25_ids:
                if bm25_docs is None:
                    # Need to reconstruct docs from Chroma
                    logger.warning("BM25 docs not found in pickle, will fetch from Chroma")
                    bm25_docs = [""] * len(bm25_ids)
                
                self.bm25_retriever = BM25Retriever(
                    bm25_obj=bm25_obj,
                    ids=bm25_ids,
                    docs=bm25_docs,
                    k=200
                )
                logger.info(f"BM25 retriever loaded with {len(bm25_ids)} documents")
        else:
            logger.warning(f"BM25 index not found at {bm25_file}, sparse retrieval disabled")
        
        # 4. Initialize reranker
        self.reranker = None
        if reranker_model:
            try:
                self.reranker = CrossEncoderRerankerWrapper(reranker_model)
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")
        
        # 5. Initialize LLM
        if self.provider == "grok":
            self.llm = GroqLLM(model=self.model_name, api_key=GROQ_API_KEY)
        elif self.provider == "gemini":
            self.llm = GeminiLLM(model=self.model_name, api_key=GEMINI_API_KEY)
        elif self.provider == "ollama":
            self.llm = OllamaLLM(model=self.model_name)
        else:
            self.llm = GroqLLM(model=self.model_name, api_key=GROQ_API_KEY)
        
        # 6. Prompt template (provider-specific for better results)
        if self.provider == "gemini":
            # More explicit prompt for Gemini to force it to use context
            self.prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="""You are Antonio Pizzi, an expert car detailing support assistant for LaCuraDellAuto.it.

MANDATORY: You MUST use the context provided below to answer the question. The context contains real customer interactions and product recommendations from Antonio, Giuseppe, and Marco. DO NOT say you need to see the car in person or that you don't have enough information - USE THE CONTEXT.

CRITICAL RULES:

1. **YOU MUST EXTRACT INFORMATION FROM THE CONTEXT:**
   - The context below contains the actual answers to similar questions
   - Find responses from "Antonio Pizzi:", "Giuseppe Pizzi:", or "Marco Fanelli:"
   - Copy their exact product recommendations and procedures
   - Present them as your own advice

2. **EXTRACT SPECIFIC PRODUCTS:**
   - Find ALL product names in the context (e.g., "Gyeon Q2M InteriorDetailer", "Labocosmetica VARIUS")
   - Include BOTH brand AND product name
   - List all products mentioned in the context

3. **INCLUDE ALL LINKS:**
   - If you see URLs like "https://www.lacuradellauto.it/..." in the context, include them
   - Format as: [Product Name](URL)
   - Include ALL links from the context

4. **EXTRACT PROCEDURES:**
   - Copy exact steps, ratios, and instructions from the context
   - Include specific numbers and measurements from the context

5. **RESPOND DIRECTLY:**
   - Answer in Italian
   - Start with a greeting, then provide the answer
   - NEVER say "contact us" or "we need to see the car" - use the context to answer
   - NEVER mention "Ticket 1" or "according to our database" - just answer naturally

ONLY say "I dati disponibili non sono sufficienti" if the context is completely empty or unrelated.

--------------------
USER QUESTION:
{question}

--------------------
CONTEXT (USE THIS TO ANSWER):
{context}

--------------------
ANSWER (in Italian, extract products and procedures from the context above):
"""
            )
        else:
            # Standard prompt for Groq/Ollama
            self.prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template="""You are Antonio, Giuseppe, or Marco - an expert car detailing support assistant for LaCuraDellAuto.it.

Answer the user's question directly and naturally, as if you are the actual support agent responding. Use the context below which contains previous customer interactions and product guides to provide accurate recommendations.

CRITICAL INSTRUCTIONS:

1. **RESPOND AS THE AGENT:**
   - Answer directly, naturally, and confidently
   - NEVER mention "Ticket 1", "Ticket 2", "according to our tickets", "in our database", "we found in our archive"
   - NEVER say "we found a similar question" or "based on previous tickets"
   - Just answer as if you know the answer from experience
   - Sound like you're giving direct advice, not referencing sources

2. **EXTRACT THE ACTUAL ANSWER:**
   - Look for responses from "Antonio Pizzi:", "Giuseppe Pizzi:", or "Marco Fanelli:" in the context
   - These contain the actual product recommendations and procedures
   - Use their exact recommendations, but present them as your own advice

3. **EXTRACT PRODUCT NAMES:**
   - Find ALL product names mentioned (e.g., "Gyeon Q2M InteriorDetailer", "Labocosmetica VARIUS", "ADBL Glass Cleaner")
   - Include BOTH brand name AND product name (e.g., "Gyeon Q2M InteriorDetailer", not just "InteriorDetailer")
   - List all recommended products clearly

4. **INCLUDE PRODUCT LINKS:**
   - If you see URLs like "https://www.lacuradellauto.it/..." in the context, ALWAYS include them
   - Format as clickable markdown links: [Product Name](URL)
   - Include ALL product links mentioned

5. **EXTRACT PROCEDURES/STEPS:**
   - If the context mentions steps (Step 1, Step 2, etc.), include them in order
   - If it mentions dilution ratios (e.g., "3-4 tappi in 10 litri"), include the exact ratio
   - If it mentions usage instructions, include them

6. **BE SPECIFIC AND DIRECT:**
   - Don't say "a product" - say the exact product name
   - Don't say "follow the procedure" - explain the actual steps
   - Include specific numbers, ratios, and measurements
   - Be confident and direct in your recommendations

7. **TONE AND STYLE:**
   - Answer in Italian
   - Use professional but friendly tone (like Antonio/Giuseppe/Marco)
   - Be helpful and specific
   - Start with a brief greeting, then get straight to the answer

ONLY say "I dati disponibili non sono sufficienti" if the context is completely unrelated to the question.

--------------------
USER QUESTION:
{question}

--------------------
CONTEXT:
{context}

--------------------
ANSWER (in Italian, respond naturally as the support agent, extract products and links from context):
"""
            )
    
    def _expand_query_terms(self, query: str) -> set:
        """Expand query with domain-specific synonyms for better matching."""
        query_lower = query.lower()
        expanded = set(query_lower.split())
        
        # Domain-specific term expansions
        expansions = {
            'ppf': ['pellicola', 'film', 'protection film', 'paint protection', 'pellicola protettiva'],
            'pellicola': ['ppf', 'film', 'protection film', 'paint protection'],
            'ingiallita': ['yellowed', 'yellowing', 'gialla', 'ingiallimento'],
            'carteggiatura': ['sanding', 'sand', 'levigare', 'levigatura'],
            'bug': ['insetto', 'insetti', 'moscerini', 'bug remover'],
            'insetti': ['bug', 'bugs', 'moscerini', 'insect'],
            'vetro': ['vetri', 'glass', 'windshield', 'parabrezza', 'cristallo'],
            'parabrezza': ['windshield', 'vetro', 'glass', 'windscreen'],
            'interni': ['interno', 'interior', 'abitacolo', 'cruscotto'],
            'lucidatura': ['polish', 'polishing', 'lucidare', 'correzione'],
        }
        
        # Add expansions for matched terms
        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded.update(synonyms)
        
        return expanded
    
    def _calculate_lexical_score(self, query_terms: set, doc_text: str) -> float:
        """Calculate lexical overlap score between query and document."""
        doc_lower = doc_text.lower()
        doc_words = set(doc_lower.split())
        
        # Count matching terms
        matches = len(query_terms & doc_words)
        
        # Boost for important domain terms
        important_terms = {'ppf', 'pellicola', 'ingiallita', 'carteggiatura', 'bug', 'vetro', 'parabrezza'}
        important_matches = len((query_terms & important_terms) & doc_words)
        
        # Score: base matches + bonus for important terms
        score = (matches * 0.1) + (important_matches * 0.3)
        return min(score, 1.0)  # Cap at 1.0
    
    def _hybrid_retrieve(
        self,
        query: str,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        top_k: int = 10
    ) -> List[Document]:
        """
        Combine dense and sparse retrieval results with hybrid scoring + lexical boosting.
        
        Args:
            query: User query
            dense_docs: Documents from dense (semantic) retrieval
            sparse_docs: Documents from sparse (BM25) retrieval
            top_k: Number of documents to return
        
        Returns:
            Ranked list of documents with hybrid scores
        """
        # Expand query terms for better matching
        query_terms = self._expand_query_terms(query)
        
        # Normalize BM25 scores
        sparse_scores = [d.metadata.get("bm25_score", 0.0) for d in sparse_docs] or [0.0]
        max_sparse = max(sparse_scores) if sparse_scores and max(sparse_scores) > 0 else 1.0
        
        # Create candidate map by document ID
        candidates: Dict[str, Dict[str, Any]] = {}
        
        # Add dense candidates
        for i, doc in enumerate(dense_docs):
            doc_id = doc.metadata.get("id") or doc.metadata.get("source_id") or f"dense_{i}"
            # Get distance from metadata or calculate from similarity
            distance = doc.metadata.get("distance")
            if distance is None:
                # Try to get from similarity_score if available
                similarity_score = doc.metadata.get("similarity_score")
                if similarity_score is not None:
                    similarity = float(similarity_score)
                else:
                    # Default: assume good match if no distance info
                    similarity = 0.8
            else:
                # Convert distance to similarity (Chroma uses cosine distance, 0=perfect, 1=no match)
                similarity = 1.0 - float(distance)
            
            # Calculate lexical score
            lexical_score = self._calculate_lexical_score(query_terms, doc.page_content)
            
            candidates[doc_id] = {
                "doc": doc,
                "dense_score": similarity,
                "sparse_score": 0.0,
                "lexical_score": lexical_score,
            }
        
        # Add/merge sparse candidates
        for i, doc in enumerate(sparse_docs):
            doc_id = doc.metadata.get("source_id") or doc.metadata.get("id") or f"sparse_{i}"
            bm25_score = doc.metadata.get("bm25_score", 0.0)
            normalized_sparse = bm25_score / max_sparse if max_sparse > 0 else 0.0
            
            # Calculate lexical score
            lexical_score = self._calculate_lexical_score(query_terms, doc.page_content)
            
            if doc_id in candidates:
                candidates[doc_id]["sparse_score"] = normalized_sparse
                # Update lexical score if higher
                candidates[doc_id]["lexical_score"] = max(candidates[doc_id]["lexical_score"], lexical_score)
            else:
                candidates[doc_id] = {
                    "doc": doc,
                    "dense_score": 0.0,
                    "sparse_score": normalized_sparse,
                    "lexical_score": lexical_score,
                }
        
        # Compute hybrid scores with lexical boosting
        scored_docs = []
        for doc_id, data in candidates.items():
            # Base hybrid score
            hybrid_score = (
                self.dense_weight * data["dense_score"] +
                self.sparse_weight * data["sparse_score"]
            )
            
            # Add lexical boost (0.2 weight for keyword matching)
            hybrid_score += 0.2 * data["lexical_score"]
            
            # Store score in metadata
            doc = data["doc"]
            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["lexical_score"] = data["lexical_score"]
            scored_docs.append((hybrid_score, doc))
        
        # Sort by hybrid score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:top_k]]
    
    def _filter_by_type(self, documents: List[Document], doc_type: str) -> List[Document]:
        """Filter documents by type (ticket or guide_chunk)."""
        return [
            doc for doc in documents
            if doc.metadata.get("type") == doc_type
        ]
    
    def retrieve(
        self,
        query: str,
        top_k_tickets: int = 3,
        top_k_guides: int = 3,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant tickets and guides using hybrid search.
        
        Returns:
            Dict with 'tickets' and 'guides' in Chroma-style format
        """
        # Get dense candidates
        if self.tickets_retriever and self.guides_retriever:
            dense_tickets = self.tickets_retriever.get_relevant_documents(query)
            dense_guides = self.guides_retriever.get_relevant_documents(query)
        else:
            # Fallback: get all and filter
            all_dense = self.general_retriever.get_relevant_documents(query)
            dense_tickets = self._filter_by_type(all_dense, "ticket")
            dense_guides = self._filter_by_type(all_dense, "guide_chunk")
        
        # Get sparse candidates
        sparse_docs = []
        if self.bm25_retriever:
            sparse_docs = self.bm25_retriever.get_relevant_documents(query)
        
        sparse_tickets = self._filter_by_type(sparse_docs, "ticket")
        sparse_guides = self._filter_by_type(sparse_docs, "guide_chunk")
        
        # Hybrid retrieval for tickets
        ticket_candidates = self._hybrid_retrieve(
            query,
            dense_tickets,
            sparse_tickets,
            top_k=max(top_k_tickets * 10, 50)  # Get more candidates for better ranking
        )
        
        # Hybrid retrieval for guides
        guide_candidates = self._hybrid_retrieve(
            query,
            dense_guides,
            sparse_guides,
            top_k=max(top_k_guides * 10, 50)
        )
        
        # Rerank if available
        if self.reranker:
            ticket_candidates = self.reranker.rerank(query, ticket_candidates, top_k=top_k_tickets * 3)
            guide_candidates = self.reranker.rerank(query, guide_candidates, top_k=top_k_guides * 3)
        
        # Build Chroma-style return structure
        def _build_chroma_format(docs: List[Document]) -> Dict[str, Any]:
            ids = []
            documents = []
            metadatas = []
            distances = []
            
            for doc in docs:
                doc_id = doc.metadata.get("id") or doc.metadata.get("source_id") or ""
                ids.append(doc_id)
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                # Convert similarity back to distance for compatibility
                hybrid_score = doc.metadata.get("hybrid_score", 0.0)
                distance = 1.0 - hybrid_score
                distances.append(distance)
            
            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
            }
        
        return {
            "query": query,
            "tickets": _build_chroma_format(ticket_candidates[:top_k_tickets]),
            "guides": _build_chroma_format(guide_candidates[:top_k_guides]),
        }
    
    def build_context(
        self,
        tickets_data: Dict[str, Any],
        guides_data: Dict[str, Any],
        max_ticket_chars: int = 1500,
        max_guide_chars: int = 1500,
        max_tickets: int = 5,
        max_guides: int = 3,
        max_total_chars: int = 15000,
    ) -> str:
        """Build readable context string from retrieved documents with size limits."""
        parts = []
        total_chars = 0
        
        if tickets_data and tickets_data.get("documents"):
            parts.append("=== HISTORICAL TICKETS ===\n")
            docs = tickets_data["documents"][0]
            metas = tickets_data["metadatas"][0]
            
            # Limit number of tickets
            for i, doc in enumerate(docs[:max_tickets]):
                if total_chars >= max_total_chars:
                    break
                    
                meta = metas[i] if i < len(metas) else {}
                ticket_id = meta.get("orig_ticket_id") or meta.get("ticket_id", "N/A")
                subject = meta.get("subject", "")
                
                header = f"[TICKET {i+1}] ID: {ticket_id}"
                if subject:
                    header += f"\nSubject: {subject}"
                header += "\n"
                
                truncated = (doc[:max_ticket_chars] + "...") if len(doc) > max_ticket_chars else doc
                ticket_text = header + truncated + "\n\n"
                
                if total_chars + len(ticket_text) > max_total_chars:
                    # Truncate this ticket to fit
                    remaining = max_total_chars - total_chars - len(header) - 10
                    if remaining > 100:
                        truncated = doc[:remaining] + "..."
                        ticket_text = header + truncated + "\n\n"
                    else:
                        break
                
                parts.append(ticket_text)
                total_chars += len(ticket_text)
        
        if guides_data and guides_data.get("documents") and total_chars < max_total_chars:
            parts.append("=== PRODUCT GUIDES ===\n")
            docs = guides_data["documents"][0]
            metas = guides_data["metadatas"][0]
            
            # Limit number of guides
            for i, doc in enumerate(docs[:max_guides]):
                if total_chars >= max_total_chars:
                    break
                    
                meta = metas[i] if i < len(metas) else {}
                guide_title = meta.get("guide_title", "")
                guide_number = meta.get("guide_number", "")
                section = meta.get("section_heading", "")
                
                header = f"[GUIDE {i+1}] {guide_title}"
                if guide_number:
                    header += f" ({guide_number})"
                if section:
                    header += f" - {section}"
                header += "\n"
                
                truncated = (doc[:max_guide_chars] + "...") if len(doc) > max_guide_chars else doc
                guide_text = header + truncated + "\n\n"
                
                if total_chars + len(guide_text) > max_total_chars:
                    # Truncate this guide to fit
                    remaining = max_total_chars - total_chars - len(header) - 10
                    if remaining > 100:
                        truncated = doc[:remaining] + "..."
                        guide_text = header + truncated + "\n\n"
                    else:
                        break
                
                parts.append(guide_text)
                total_chars += len(guide_text)
        
        return "\n".join(parts)
    
    def answer(
        self,
        query: str,
        top_k_tickets: int = 3,
        top_k_guides: int = 3,
    ) -> Dict[str, Any]:
        """
        Main answer method - retrieves context and generates answer.
        
        Returns:
            Dict with 'query', 'answer', 'context', 'sources'
        """
        # Retrieve
        retrieved = self.retrieve(query, top_k_tickets, top_k_guides)
        
        # Build context
        context = self.build_context(
            retrieved.get("tickets"),
            retrieved.get("guides")
        )
        
        # Build prompt
        prompt = self.prompt_template.format(question=query, context=context)
        
        # Generate answer
        answer = self.llm(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": retrieved,
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about indexed documents."""
        try:
            # Try to get collection
            collection = self.chroma._collection
            if collection:
                count = collection.count()
                return {"tickets": count, "guides": count}  # Approximate
        except Exception:
            pass
        
        return {"tickets": 0, "guides": 0}
