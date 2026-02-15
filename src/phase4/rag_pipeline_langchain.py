"""
LangChain-powered RAG pipeline - Optimized & Refactored
===================================================

This module orchestrates the RAG components:
- Embeddings & VectorStore (Chroma)
- Sparse Retrieval (BM25)
- Hybrid Scoring
- LLM interaction
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Internal modules
from config.settings import (
    CHROMA_DB_DIR,
    LOCAL_EMBEDDING_MODEL,
    BM25_INDEX_PATH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RERANKER_MODEL,
)
from src.phase4.llm import get_llm
from src.phase4.prompts import ANTONIO_PROMPT
from src.phase4.retrieval import (
    BM25Retriever,
    CrossEncoderRerankerWrapper,
    hybrid_score_documents,
)

logger = logging.getLogger(__name__)

class LangchainRAG:
    """Orchestrator for the LangChain RAG pipeline."""
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = None,
        chroma_dir: Optional[Path] = None,
        hf_model_name: Optional[str] = None,
        bm25_path: Optional[Path] = None,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        sparse_weight: float = HYBRID_SPARSE_WEIGHT,
    ):
        self.provider = provider.lower()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # 1. Initialize LLM
        self.llm = get_llm(self.provider, model)
        self.prompt_template = ANTONIO_PROMPT
        
        # 2. Initialize Embeddings
        hf_model = hf_model_name or LOCAL_EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 3. Initialize Chroma
        chroma_path = chroma_dir or CHROMA_DB_DIR
        self.chroma = Chroma(
            collection_name="rag_v2",
            embedding_function=self.embeddings,
            persist_directory=str(chroma_path),
        )
        
        # 4. Initialize BM25
        bm25_file = bm25_path or BM25_INDEX_PATH
        self.bm25_retriever = None
        if bm25_file.exists():
            with open(bm25_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.bm25_retriever = BM25Retriever(
                    bm25_obj=data.get("bm25"),
                    ids=data.get("ids", []),
                    docs=data.get("corpus", data.get("docs", [])),  # Support both keys
                    metadatas=data.get("metadatas", []),  # Pass metadata
                    k=50
                )
        
        # 5. Initialize Reranker (Optional)
        self.reranker = CrossEncoderRerankerWrapper(RERANKER_MODEL) if RERANKER_MODEL else None

    def _safe_search(self, query: str, k: int, filter_dict: dict) -> list:
        """Safely search ChromaDB, returning [] on any error."""
        try:
            results = self.chroma.similarity_search_with_score(query, k=k, filter=filter_dict)
            for doc, score in results:
                doc.metadata["distance"] = float(score)
            return results
        except Exception as e:
            logger.warning(f"Dense search failed for filter {filter_dict}: {e}")
            return []

    def retrieve(
        self,
        query: str,
        top_k_tickets: int = 3,
        top_k_guides: int = 3,
        top_k_qa: int = 5,  # Prioritize Q&A pairs
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """Hybrid retrieval for tickets, guides, and Q&A pairs."""
        # Dense retrieval for each type (safe - won't crash on corrupted DB)
        dense_tickets = self._safe_search(query, 50, {"type": "ticket"})
        dense_guides = self._safe_search(query, 50, {"type": "guide_chunk"})
        dense_qa = self._safe_search(query, 50, {"type": "qa_pair"})
        
        dense_tickets_docs = [d for d, _ in dense_tickets]
        dense_guides_docs = [d for d, _ in dense_guides]
        dense_qa_docs = [d for d, _ in dense_qa]
        
        if fast_mode or not self.bm25_retriever:
            return {
                "tickets": self._to_chroma_format(dense_tickets_docs[:top_k_tickets]),
                "guides": self._to_chroma_format(dense_guides_docs[:top_k_guides]),
                "qa_pairs": self._to_chroma_format(dense_qa_docs[:top_k_qa]),
            }
        
        # Sparse retrieval
        sparse_docs = self.bm25_retriever.get_relevant_documents(query)
        sparse_tickets = [d for d in sparse_docs if d.metadata.get("type") == "ticket"]
        sparse_guides = [d for d in sparse_docs if d.metadata.get("type") == "guide_chunk"]
        sparse_qa = [d for d in sparse_docs if d.metadata.get("type") == "qa_pair"]
        
        # Hybrid scoring for each type
        ticket_candidates = hybrid_score_documents(
            query, dense_tickets_docs, sparse_tickets, self.dense_weight, self.sparse_weight, top_k=20
        )
        guide_candidates = hybrid_score_documents(
            query, dense_guides_docs, sparse_guides, self.dense_weight, self.sparse_weight, top_k=20
        )
        qa_candidates = hybrid_score_documents(
            query, dense_qa_docs, sparse_qa, self.dense_weight, self.sparse_weight, top_k=20
        )
        
        # Hydrate metadata for BM25-only results
        ticket_candidates = self._hydrate_metadata(ticket_candidates)
        guide_candidates = self._hydrate_metadata(guide_candidates)
        qa_candidates = self._hydrate_metadata(qa_candidates)
        
        return {
            "query": query,
            "tickets": self._to_chroma_format(ticket_candidates[:top_k_tickets]),
            "guides": self._to_chroma_format(guide_candidates[:top_k_guides]),
            "qa_pairs": self._to_chroma_format(qa_candidates[:top_k_qa]),
        }

    def _hydrate_metadata(self, docs: List[Document]) -> List[Document]:
        """Fetch full metadata from Chroma collection for documents missing it."""
        def _sid(d):
            return d.metadata.get("source_id") or d.metadata.get("id")
        missing_ids = [_sid(d) for d in docs if "subject" not in d.metadata and _sid(d)]
        if not missing_ids: return docs
        
        try:
            # Fetch one by one to avoid "Error finding id" on missing IDs
            id_to_meta = {}
            for mid in missing_ids:
                try:
                    result = self.chroma._collection.get(ids=[mid])
                    if result and result.get("ids"):
                        for i, rid in enumerate(result["ids"]):
                            id_to_meta[rid] = result["metadatas"][i]
                except Exception:
                    continue  # Skip IDs that don't exist in Chroma
            
            for doc in docs:
                sid = _sid(doc)
                if sid and sid in id_to_meta:
                    preserved = {k: v for k, v in doc.metadata.items() if k in ["hybrid_score", "distance", "lexical_score"]}
                    doc.metadata.update(id_to_meta[sid])
                    doc.metadata.update(preserved)
        except Exception as e:
            logger.warning(f"Metadata hydration failed: {e}")
        return docs

    def _to_chroma_format(self, docs: List[Document]) -> Dict[str, Any]:
        """Helper to convert docs to UI-friendly format."""
        def _doc_id(d):
            return d.metadata.get("source_id") or d.metadata.get("id") or ""
        return {
            "ids": [[_doc_id(d) for d in docs]],
            "documents": [[d.page_content for d in docs]],
            "metadatas": [[d.metadata for d in docs]],
            "distances": [[float(d.metadata.get("distance", 0.5)) for d in docs]]
        }

    def _preserve_urls_in_truncation(self, text: str, max_chars: int) -> str:
        """Truncate text while ensuring we don't cut in the middle of a URL."""
        if len(text) <= max_chars: return text
        
        truncated = text[:max_chars]
        # Look for partial URLs at the end
        last_https = truncated.rfind("https://")
        if last_https != -1:
            # Check if it was likely cut off (no space or newline after https://)
            tail = truncated[last_https:]
            if " " not in tail and "\n" not in tail:
                # Find the end of the URL in the original text
                original_tail = text[last_https:]
                end_match = re.search(r"[\s\n]", original_tail)
                if end_match:
                    url_end = last_https + end_match.start()
                    # If the full URL fits, take it. Otherwise, cut before the URL starts
                    if url_end <= max_chars + 100: # Small buffer
                        return text[:url_end]
                    else:
                        return text[:last_https].rstrip()
        return truncated.rstrip()

    def build_context(self, tickets_data: Dict[str, Any], guides_data: Dict[str, Any], qa_pairs_data: Dict[str, Any] = None) -> str:
        """Build context string with safety character limits. Prioritizes Q&A pairs."""
        parts = []
        total_chars = 0
        MAX_TOTAL = 25000
        MAX_QA = 2000  # Q&A pairs are focused, don't need much space
        MAX_TICKET = 10000
        
        # FIRST: Process Q&A Pairs (most relevant, focused answers)
        if qa_pairs_data and qa_pairs_data.get("documents") and qa_pairs_data["documents"][0]:
            parts.append("=== RELEVANT Q&A FROM OUR SUPPORT TEAM ===")
            for i, doc in enumerate(qa_pairs_data["documents"][0]):
                meta = qa_pairs_data["metadatas"][0][i]
                header = f"\n[Q&A {i+1}] From Ticket {meta.get('orig_ticket_id', 'N/A')}\n"
                # Q&A pairs are smaller, less truncation needed
                body = self._preserve_urls_in_truncation(doc, MAX_QA)
                text = header + body + "\n"
                
                if total_chars + len(text) > MAX_TOTAL: break
                parts.append(text)
                total_chars += len(text)
        
        # SECOND: Process Tickets (fallback/supplementary)
        if tickets_data.get("documents") and tickets_data["documents"][0] and total_chars < MAX_TOTAL:
            parts.append("\n=== HISTORICAL TICKETS ===")
            for i, doc in enumerate(tickets_data["documents"][0]):
                meta = tickets_data["metadatas"][0][i]
                header = f"\n[TICKET {i+1}] ID: {meta.get('orig_ticket_id', 'N/A')} Subject: {meta.get('subject', '')}\n"
                # Use URL-safe truncation
                body = self._preserve_urls_in_truncation(doc, MAX_TICKET)
                text = header + body + "\n"
                
                if total_chars + len(text) > MAX_TOTAL: break
                parts.append(text)
                total_chars += len(text)
                
        # THIRD: Process Guides
        if guides_data.get("documents") and guides_data["documents"][0] and total_chars < MAX_TOTAL:
            parts.append("\n=== PRODUCT GUIDES ===")
            for i, doc in enumerate(guides_data["documents"][0]):
                meta = guides_data["metadatas"][0][i]
                header = f"\n[GUIDE {i+1}] {meta.get('guide_title', '')}\n"
                body = self._preserve_urls_in_truncation(doc, 3000)
                text = header + body + "\n"
                
                if total_chars + len(text) > MAX_TOTAL: break
                parts.append(text)
                total_chars += len(text)
                
        return "\n".join(parts)

    def _extract_links(self, context: str) -> str:
        """Extract all unique product/video links from the context for the LLM."""
        # Find markdown links [Name](URL) or raw URLs
        md_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', context)
        raw_links = re.findall(r'(https://www\.lacuradellauto\.it/[^\s\n\(\)\]]+)', context)
        yt_links = re.findall(r'(https://www\.youtube\.com/[^\s\n\(\)\]]+)', context)
        
        seen = set()
        links_list = []
        
        # Add MD links first
        for name, url in md_links:
            if url not in seen:
                links_list.append(f"- {name}: {url}")
                seen.add(url)
        
        # Add YT links
        for url in yt_links:
            if url not in seen:
                links_list.append(f"- Video: {url}")
                seen.add(url)
                
        # Add raw product links
        for url in raw_links:
            if url not in seen:
                # Try to find a name before it (very simple heuristic)
                links_list.append(f"- Prodotto: {url}")
                seen.add(url)
                
        return "\n".join(links_list) if links_list else "Nessun link disponibile."

    def answer(self, query: str, **kwargs) -> Dict[str, Any]:
        """Main RAG flow: Retrieve -> Context -> LLM."""
        retrieved = self.retrieve(query, **kwargs)
        # Pass Q&A pairs to build_context (they're prioritized)
        context = self.build_context(
            retrieved["tickets"], 
            retrieved["guides"],
            retrieved.get("qa_pairs")  # New: Q&A pairs for focused answers
        )
        
        # NEW: Extract links to help the LLM format them correctly
        links_block = self._extract_links(context)
        
        prompt = self.prompt_template.format(
            question=query, 
            context=context,
            links=links_block
        )
        
        answer = self.llm(prompt)
        return {"query": query, "answer": answer, "context": context, "sources": retrieved}

    def get_stats(self) -> Dict[str, int]:
        """Aggregate counts from vector store."""
        try:
            count = self.chroma._collection.count()
            if count == 0:
                return {"tickets": 0, "guides": 0, "qa_pairs": 0}
            metas = self.chroma._collection.get(limit=15000, include=["metadatas"])["metadatas"]
            return {
                "tickets": sum(1 for m in metas if m.get("type") == "ticket"),
                "guides": sum(1 for m in metas if m.get("type") == "guide_chunk"),
                "qa_pairs": sum(1 for m in metas if m.get("type") == "qa_pair"),
            }
        except Exception:
            return {"tickets": 0, "guides": 0, "qa_pairs": 0}
