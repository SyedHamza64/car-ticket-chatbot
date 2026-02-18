"""
Retrieval components for the RAG pipeline.
Includes BM25, CrossEncoder reranking, and hybrid scoring.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from config.settings import RERANKER_MODEL

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """LangChain-compatible BM25 retriever using existing pickle index."""
    
    def __init__(self, bm25_obj: BM25Okapi, ids: List[str], docs: List[str], k: int = 50, metadatas: List[Dict] = None):
        # Set attributes directly
        object.__setattr__(self, 'bm25', bm25_obj)
        object.__setattr__(self, 'ids', ids)
        object.__setattr__(self, 'docs', docs)
        object.__setattr__(self, 'metadatas', metadatas or [])  # Store metadatas from ChromaDB
        object.__setattr__(self, 'k', k)
        # Add attributes required by BaseRetriever
        object.__setattr__(self, 'tags', [])
        object.__setattr__(self, 'metadata', {})
    
    @staticmethod
    def _infer_type_from_id(doc_id: str) -> str:
        """Infer document type from its ID pattern when metadata is missing."""
        if not doc_id:
            return ""
        if "_qa_" in doc_id:
            return "qa_pair"
        if doc_id.startswith("ticket_"):
            return "ticket"
        if "__sec" in doc_id or "__chunk" in doc_id:
            return "guide_chunk"
        return ""

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
                
                # Start with stored metadata from ChromaDB if available
                if idx < len(self.metadatas) and self.metadatas[idx]:
                    metadata = dict(self.metadatas[idx])  # Copy to avoid mutation
                else:
                    metadata = {}
                
                # Ensure 'type' is always set (infer from ID if missing)
                if not metadata.get("type"):
                    metadata["type"] = self._infer_type_from_id(doc_id)
                
                # Add BM25-specific fields
                metadata["bm25_score"] = float(score)
                metadata["source_id"] = doc_id
                
                documents.append(Document(page_content=doc_text, metadata=metadata))
        
        return documents


class CrossEncoderRerankerWrapper:
    """Wrapper for CrossEncoder reranking compatible with LangChain."""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        try:
            device = "cpu" if not os.getenv("USE_CUDA", "").lower() == "true" else None
            self.ce = CrossEncoder(model_name, device=device)
            logger.info(f"Loaded CrossEncoder reranker: {model_name}" + (" (CPU)" if device else ""))
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


def expand_query_terms(query: str) -> set:
    """Expand query with domain-specific synonyms."""
    query_lower = query.lower()
    expanded = set(query_lower.split())
    
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
    
    for term, synonyms in expansions.items():
        if term in query_lower:
            expanded.update(synonyms)
    
    return expanded


def calculate_lexical_score(query_terms: set, doc_text: str) -> float:
    """Calculate lexical overlap score."""
    doc_lower = doc_text.lower()
    doc_words = set(doc_lower.split())
    matches = len(query_terms & doc_words)
    
    important_terms = {'ppf', 'pellicola', 'ingiallita', 'carteggiatura', 'bug', 'vetro', 'parabrezza'}
    important_matches = len((query_terms & important_terms) & doc_words)
    
    score = (matches * 0.1) + (important_matches * 0.3)
    return min(score, 1.0)


def hybrid_score_documents(
    query: str,
    dense_docs: List[Document],
    sparse_docs: List[Document],
    dense_weight: float,
    sparse_weight: float,
    top_k: int = 10
) -> List[Document]:
    """Combines dense and sparse results with hybrid scoring and lexical boosting."""
    query_terms = expand_query_terms(query)
    
    sparse_scores = [d.metadata.get("bm25_score", 0.0) for d in sparse_docs] or [0.0]
    max_sparse = max(sparse_scores) if sparse_scores and max(sparse_scores) > 0 else 1.0
    
    candidates: Dict[str, Dict[str, Any]] = {}
    
    # Add dense
    for i, doc in enumerate(dense_docs):
        doc_id = doc.metadata.get("id") or doc.metadata.get("source_id") or f"dense_{i}"
        distance = doc.metadata.get("distance", 0.5)
        similarity = 1.0 - float(distance)
        lexical_score = calculate_lexical_score(query_terms, doc.page_content)
        
        candidates[doc_id] = {
            "doc": doc,
            "dense_score": similarity,
            "sparse_score": 0.0,
            "lexical_score": lexical_score,
        }
    
    # Add sparse
    for i, doc in enumerate(sparse_docs):
        doc_id = doc.metadata.get("source_id") or doc.metadata.get("id") or f"sparse_{i}"
        bm25_score = doc.metadata.get("bm25_score", 0.0)
        normalized_sparse = bm25_score / max_sparse if max_sparse > 0 else 0.0
        lexical_score = calculate_lexical_score(query_terms, doc.page_content)
        
        if doc_id in candidates:
            candidates[doc_id]["sparse_score"] = normalized_sparse
            candidates[doc_id]["lexical_score"] = max(candidates[doc_id]["lexical_score"], lexical_score)
        else:
            candidates[doc_id] = {
                "doc": doc,
                "dense_score": 0.0,
                "sparse_score": normalized_sparse,
                "lexical_score": lexical_score,
            }
    
    scored_docs = []
    for doc_id, data in candidates.items():
        h_score = (dense_weight * data["dense_score"] + sparse_weight * data["sparse_score"])
        h_score += 0.4 * data["lexical_score"]  # Lexical boost
        
        final_score = min(1.0, h_score / 1.3)
        doc = data["doc"]
        doc.metadata["hybrid_score"] = final_score
        doc.metadata["distance"] = 1.0 - final_score
        scored_docs.append((final_score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
