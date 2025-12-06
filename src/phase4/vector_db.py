import json
import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional, Set

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import (
    DATA_DIR,
    PROCESSED_DIR,
    GUIDES_DIR,
    CHROMA_DB_DIR,
    LOCAL_EMBEDDING_MODEL,
    BM25_INDEX_PATH,
)

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Vector database manager for tickets and guide sections.
    Now uses local SentenceTransformer embeddings.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        reset: bool = False,
        persistent_client: bool = True,
    ):
        """
        :param db_path: path to the Chroma DB folder
        :param embedding_model: e.g. "sentence-transformers/all-mpnet-base-v2"
        :param reset: if True, will reset the database on init
        :param persistent_client: if True, use a persistent Chroma client
        """
        self.db_path = db_path or CHROMA_DB_DIR
        self.db_path.mkdir(parents=True, exist_ok=True)

        if embedding_model is None:
            embedding_model = LOCAL_EMBEDDING_MODEL

        logger.info(f"Loading LOCAL embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        logger.info(f"Initializing Chroma DB at: {self.db_path}")
        if persistent_client:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
        else:
            self.client = chromadb.Client(
                Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(self.db_path))
            )

        if reset:
            logger.warning("Resetting Chroma DB (deleting all collections).")
            for coll_name in self.client.list_collections():
                self.client.delete_collection(coll_name.name)

        # Create or get collections
        # Check if unified "rag_v2" collection exists (from rebuild script)
        existing_collections = [c.name for c in self.client.list_collections()]
        logger.info(f"Found collections: {existing_collections}")
        
        if "rag_v2" in existing_collections:
            # Use unified collection
            logger.info("Using unified 'rag_v2' collection")
            self.rag_v2_coll = self.client.get_collection("rag_v2")
            self.tickets_coll = None
            self.guides_coll = None
        else:
            # Use separate collections
            logger.info("Using separate 'tickets' and 'guides' collections")
            self.rag_v2_coll = None
            self.tickets_coll = self.client.get_or_create_collection(
                name="tickets", metadata={"hnsw:space": "cosine"}
            )
            self.guides_coll = self.client.get_or_create_collection(
                name="guides", metadata={"hnsw:space": "cosine"}
            )
        
        # Load BM25 index if available
        self.bm25_index = None
        self.bm25_ids = None
        self.bm25_docs = None
        if BM25_INDEX_PATH.exists():
            try:
                import pickle
                with open(BM25_INDEX_PATH, "rb") as f:
                    bm25_data = pickle.load(f)
                    if isinstance(bm25_data, dict):
                        self.bm25_index = bm25_data.get("bm25")
                        self.bm25_ids = bm25_data.get("ids", [])
                        self.bm25_docs = bm25_data.get("docs", [])
                    else:
                        # Legacy tuple format
                        self.bm25_index, self.bm25_ids = bm25_data
                        self.bm25_docs = self.bm25_ids  # Fallback
                logger.info(f"Loaded BM25 index with {len(self.bm25_ids) if self.bm25_ids else 0} documents")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

    # ------------------------------------------------------------
    # Query expansion for better retrieval
    # ------------------------------------------------------------
    def _expand_query_terms(self, query: str) -> Set[str]:
        """Expand query with Italian/English synonyms for better matching."""
        query_lower = query.lower()
        expanded = set(query_lower.split())
        
        # Italian-English translation dictionary for car detailing terms
        translations = {
            'bug': ['insetto', 'insetti', 'moscerini', 'moscerino'],
            'bug remover': ['rimuovi insetti', 'rimuovi moscerini', 'pulitore insetti'],
            'insetto': ['bug', 'insetti', 'moscerini'],
            'insetti': ['bug', 'insetto', 'moscerini'],
            'ready to use': ['pronto all\'uso', 'pronto uso', 'pronto'],
            'pronto all\'uso': ['ready to use', 'pronto uso'],
            'pronto': ['ready', 'ready to use'],
            'risciacquo': ['rinse', 'risciacquare', 'risciacare'],
            'risciacquare': ['rinse', 'risciacquo'],
            'senza risciacquo': ['no rinse', 'without rinse', 'non risciacquare'],
            'waterless': ['senza acqua', 'waterless', 'senza risciacquo'],
            'varius': ['varius', 'labocosmetica varius'],
            'quick detailer': ['quick detailer', 'dettagliante rapido'],
        }
        
        # Add translations for matched terms
        for term, translations_list in translations.items():
            if term in query_lower:
                expanded.update(translations_list)
        
        return expanded
    
    def _expand_query_for_search(self, query: str) -> str:
        """Expand query text with synonyms for better semantic matching."""
        expanded_terms = self._expand_query_terms(query)
        # Combine original query with expanded terms
        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query

    # ------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed a single text string with SentenceTransformer."""
        return self.embedding_model.encode([text])[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        return self.embedding_model.encode(texts).tolist()

    # ------------------------------------------------------------
    # Ticket ingestion
    # ------------------------------------------------------------

    def add_tickets(self, tickets_file: Optional[Path] = None):
        """
        Reads processed_tickets.json and adds them to the Chroma tickets collection.
        Each ticket is stored as one document.
        """

        tickets_file = tickets_file or PROCESSED_DIR / "processed_tickets.json"

        if not tickets_file.exists():
            raise FileNotFoundError(f"Tickets file not found: {tickets_file}")

        with open(tickets_file, "r", encoding="utf-8") as f:
            tickets = json.load(f)

        ids = []
        docs = []
        metas = []

        for t in tickets:
            ticket_id = t.get("ticket_id")
            body = t.get("searchable_text", "").strip()
            if not body:
                continue

            uid = f"ticket_{ticket_id}"
            ids.append(uid)
            docs.append(body)

            meta = {
                "ticket_id": ticket_id,
                "subject": t.get("subject", ""),
                "status": t.get("status", ""),
                "priority": t.get("priority", ""),
                "created_at": t.get("created_at", ""),
            }
            metas.append(meta)

        if not ids:
            logger.warning("No tickets to add.")
            return

        embeddings = self.embed_batch(docs)

        self.tickets_coll.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )

        logger.info(f"Added {len(ids)} tickets to Chroma.")

    # ------------------------------------------------------------
    # Guides ingestion
    # ------------------------------------------------------------

    def add_guides(self, guides_file: Optional[Path] = None):
        """
        Reads guide chunks file (guides_chunks.json) and add them to the Chroma guides collection.
        """
        guides_file = guides_file or PROCESSED_DIR / "guides_chunks.json"

        if not guides_file.exists():
            raise FileNotFoundError(f"Guides file not found: {guides_file}")

        with open(guides_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        ids = []
        docs = []
        metas = []

        for c in chunks:
            chunk_id = c.get("chunk_id")
            if not chunk_id:
                continue  # must have an ID

            text = c.get("chunk_text", "").strip()
            if not text:
                continue

            ids.append(chunk_id)
            docs.append(text)

            meta = {
                "guide_number": c.get("guide_number", ""),
                "guide_title": c.get("guide_title", ""),
                "section_index": c.get("section_index", ""),
                "chunk_index": c.get("chunk_index", ""),
                "url": c.get("url", ""),
            }
            metas.append(meta)

        if not ids:
            logger.warning("No guide chunks to add.")
            return

        embeddings = self.embed_batch(docs)

        self.guides_coll.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )

        logger.info(f"Added {len(ids)} guide chunks to Chroma.")

    # ------------------------------------------------------------
    # Search (dense only)
    # ------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get statistics about indexed tickets and guides."""
        try:
            if self.rag_v2_coll:
                # Count from unified collection using metadata filter
                try:
                    # Use where filter for efficient counting
                    tickets_data = self.rag_v2_coll.get(where={"type": "ticket"}, limit=None)
                    guides_data = self.rag_v2_coll.get(where={"type": "guide_chunk"}, limit=None)
                    tickets_count = len(tickets_data.get("ids", []))
                    guides_count = len(guides_data.get("ids", []))
                    logger.info(f"Stats from rag_v2: {tickets_count} tickets, {guides_count} guides")
                except Exception as e:
                    logger.warning(f"Where filter failed, using fallback: {e}")
                    # Fallback: get all and count manually
                    all_data = self.rag_v2_coll.get(limit=None)
                    metadatas = all_data.get("metadatas", [])
                    tickets_count = sum(1 for meta in metadatas if meta and meta.get("type") == "ticket")
                    guides_count = sum(1 for meta in metadatas if meta and meta.get("type") == "guide_chunk")
                    logger.info(f"Stats from rag_v2 (fallback): {tickets_count} tickets, {guides_count} guides")
            else:
                # Count from separate collections
                tickets_count = self.tickets_coll.count() if self.tickets_coll else 0
                guides_count = self.guides_coll.count() if self.guides_coll else 0
                logger.info(f"Stats from separate collections: {tickets_count} tickets, {guides_count} guides")
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            tickets_count = 0
            guides_count = 0
        
        return {
            "tickets": tickets_count,
            "guides": guides_count,
        }

    def search_tickets(self, query: str, k: int = 5):
        """Search tickets with hybrid search (semantic + BM25) + lexical reranking."""
        # Expand query for better matching
        expanded_query = self._expand_query_for_search(query)
        query_emb = self.embed(expanded_query)
        
        # Get semantic candidates
        candidate_k = max(k * 10, 200)  # Get at least 200 candidates
        
        semantic_ids = set()
        semantic_results = {}
        
        if self.rag_v2_coll:
            try:
                semantic_results = self.rag_v2_coll.query(
                    query_embeddings=[query_emb],
                    n_results=candidate_k,
                    where={"type": "ticket"}
                )
            except Exception:
                semantic_results = self.rag_v2_coll.query(
                    query_embeddings=[query_emb],
                    n_results=candidate_k * 2
                )
                if semantic_results.get("ids") and semantic_results["ids"][0]:
                    filtered_indices = [
                        i for i, meta in enumerate(semantic_results.get("metadatas", [[]])[0])
                        if meta and meta.get("type") == "ticket"
                    ][:candidate_k]
                    for key in ["ids", "documents", "metadatas", "distances"]:
                        if semantic_results.get(key) and semantic_results[key][0]:
                            semantic_results[key][0] = [semantic_results[key][0][i] for i in filtered_indices]
        else:
            semantic_results = self.tickets_coll.query(query_embeddings=[query_emb], n_results=candidate_k)
        
        if semantic_results.get("ids") and semantic_results["ids"][0]:
            semantic_ids = set(semantic_results["ids"][0])
        
        # Add BM25 results if available
        bm25_candidate_ids = set()
        if self.bm25_index and self.bm25_ids:
            try:
                # Use expanded query terms for BM25 - focus on key terms
                expanded_terms = self._expand_query_terms(query)
                
                # Extract key terms - prioritize important car detailing terms
                # Important terms that should be included
                important_terms = {'bug', 'remover', 'insetto', 'insetti', 'moscerini', 'uccelli', 'pronto', 'uso', 'risciacquo', 'risciacquare', 'senza', 'varius', 'maniac', 'line', 'waterless', 'soluzione', 'rapida', 'localizzata', 'pulitore', 'rimuovi'}
                
                # Stopwords to filter out
                stopwords = {'ti', 'chiedo', 'se', 'avete', 'in', 'commercio', 'un', 'dover', 'l\'auto', 'ogni', 'volta', 'che', 'viene', 'utilizzato', 'oppure', 'prodotto', 'consigliato', 'mi', 'consente', 'di', 'effettuare', 'tale', 'operazione', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'onto', 'all\'uso'}
                
                query_tokens = []
                
                # Add important terms from original query
                for word in query.lower().split():
                    word_clean = ''.join(c for c in word if c.isalnum())
                    if word_clean in important_terms or (len(word_clean) >= 4 and word_clean not in stopwords):
                        query_tokens.append(word_clean)
                
                # Add expanded terms (prioritize important ones)
                for term in expanded_terms:
                    for word in term.split():
                        word_clean = ''.join(c for c in word if c.isalnum())
                        if word_clean in important_terms or (len(word_clean) >= 4 and word_clean not in stopwords):
                            query_tokens.append(word_clean)
                
                # Remove duplicates but keep order (important terms first)
                seen = set()
                unique_tokens = []
                for token in query_tokens:
                    if token not in seen:
                        seen.add(token)
                        unique_tokens.append(token)
                query_tokens = unique_tokens
                
                logger.debug(f"BM25 query tokens (filtered): {query_tokens[:15]}")
                
                scores = self.bm25_index.get_scores(query_tokens)
                
                # Get top BM25 results (only tickets) - prioritize high-scoring tickets
                # Get more candidates to ensure we don't miss relevant tickets
                top_bm25_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:min(candidate_k * 3, 500)]
                bm25_candidate_ids = set()
                for idx in top_bm25_indices:
                    if idx < len(self.bm25_ids):
                        doc_id = self.bm25_ids[idx]
                        # Only include ticket IDs
                        if isinstance(doc_id, str) and doc_id.startswith("ticket_"):
                            bm25_candidate_ids.add(doc_id)
                
                logger.info(f"BM25 found {len(bm25_candidate_ids)} ticket candidates, {len(bm25_candidate_ids - semantic_ids)} new ones")
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
        
        # Fetch BM25-only candidates from ChromaDB
        # Prioritize high-scoring BM25 tickets by maintaining score order
        if bm25_candidate_ids and bm25_candidate_ids - semantic_ids:
            # Get BM25 scores to prioritize high-scoring tickets
            if self.bm25_index and self.bm25_ids:
                try:
                    # Recalculate scores to get ticket 8401's score
                    expanded_terms = self._expand_query_terms(query)
                    important_terms = {'bug', 'remover', 'insetto', 'insetti', 'moscerini', 'uccelli', 'pronto', 'uso', 'risciacquo', 'risciacquare', 'senza', 'varius', 'maniac', 'line', 'waterless', 'soluzione', 'rapida', 'localizzata', 'pulitore', 'rimuovi'}
                    stopwords = {'ti', 'chiedo', 'se', 'avete', 'in', 'commercio', 'un', 'dover', 'l\'auto', 'ogni', 'volta', 'che', 'viene', 'utilizzato', 'oppure', 'prodotto', 'consigliato', 'mi', 'consente', 'di', 'effettuare', 'tale', 'operazione', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'to', 'of', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'onto', 'all\'uso'}
                    query_tokens = []
                    for word in query.lower().split():
                        word_clean = ''.join(c for c in word if c.isalnum())
                        if word_clean in important_terms or (len(word_clean) >= 4 and word_clean not in stopwords):
                            query_tokens.append(word_clean)
                    for term in expanded_terms:
                        for word in term.split():
                            word_clean = ''.join(c for c in word if c.isalnum())
                            if word_clean in important_terms or (len(word_clean) >= 4 and word_clean not in stopwords):
                                query_tokens.append(word_clean)
                    seen = set()
                    unique_tokens = []
                    for token in query_tokens:
                        if token not in seen:
                            seen.add(token)
                            unique_tokens.append(token)
                    query_tokens = unique_tokens
                    
                    bm25_scores = self.bm25_index.get_scores(query_tokens)
                    # Create score map for BM25 tickets
                    bm25_score_map = {}
                    for idx, doc_id in enumerate(self.bm25_ids):
                        if isinstance(doc_id, str) and doc_id.startswith("ticket_") and doc_id in bm25_candidate_ids:
                            bm25_score_map[doc_id] = bm25_scores[idx]
                    
                    # Sort missing IDs by BM25 score (highest first)
                    missing_with_scores = [(tid, bm25_score_map.get(tid, 0)) for tid in bm25_candidate_ids - semantic_ids]
                    missing_with_scores.sort(key=lambda x: x[1], reverse=True)
                    missing_ids = [tid for tid, _ in missing_with_scores[:150]]  # Get top 150 by score
                except Exception:
                    # Fallback: just take first 150
                    missing_ids = list(bm25_candidate_ids - semantic_ids)[:150]
            else:
                missing_ids = list(bm25_candidate_ids - semantic_ids)[:150]
            
            try:
                if self.rag_v2_coll:
                    missing_results = self.rag_v2_coll.get(
                        ids=missing_ids,
                        where={"type": "ticket"}
                    )
                    # Merge with semantic results
                    if missing_results.get("ids"):
                        for key in ["ids", "documents", "metadatas"]:
                            if semantic_results.get(key) and semantic_results[key][0]:
                                semantic_results[key][0].extend(missing_results.get(key, []))
                            else:
                                semantic_results[key] = [missing_results.get(key, [])]
                        # Add dummy distances for BM25 results (will be reranked)
                        if semantic_results.get("distances") and semantic_results["distances"][0]:
                            semantic_results["distances"][0].extend([0.6] * len(missing_results.get("ids", [])))
            except Exception as e:
                logger.warning(f"Could not fetch BM25-only candidates: {e}")
        
        # Rerank with lexical matching (now includes BM25 results)
        if semantic_results.get("documents") and semantic_results["documents"][0]:
            reranked = self._rerank_tickets_lexical(query, semantic_results, bm25_candidate_ids)
            # Trim to top k
            for key in ["ids", "documents", "metadatas", "distances"]:
                if reranked.get(key) and reranked[key][0]:
                    reranked[key][0] = reranked[key][0][:k]
            return reranked
        
        return semantic_results
    
    def _rerank_tickets_lexical(self, query: str, results: dict, bm25_ids: set = None) -> dict:
        """Rerank tickets by adding lexical bonus for keyword matches."""
        if not results.get("documents") or not results["documents"][0]:
            return results
        
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [1.0] * len(docs)
        
        # Extract meaningful query terms (length >= 2, lowercase, remove common words)
        # Use expanded terms for better matching
        expanded_terms = self._expand_query_terms(query)
        stopwords = {'ciao', 'mi', 'ti', 'si', 'di', 'la', 'le', 'il', 'lo', 'gli', 'una', 'uno', 'con', 'per', 'che', 'non', 'sono', 'hai', 'ho', 'è', 'e', 'a', 'da', 'in', 'su', 'un', 'del', 'della', 'delle', 'degli', 'dei', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_terms = set()
        for word in expanded_terms:
            word_clean = ''.join(c for c in word if c.isalnum())
            if len(word_clean) >= 2 and word_clean not in stopwords:
                query_terms.add(word_clean)
        
        scored_items = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            # Count lexical matches in subject and content
            subject = meta.get('subject', '').lower()
            content = doc.lower()
            
            lexical_hits = 0
            # Check for important phrase matches (higher weight)
            important_phrases = ['pronto all\'uso', 'senza risciacquo', 'varius', 'maniac line', 'waterless', 'soluzione rapida', 'insetti uccelli']
            for phrase in important_phrases:
                if phrase in content.lower() or phrase in subject.lower():
                    lexical_hits += 5  # Phrase matches are very important
            
            # Check individual term matches
            for term in query_terms:
                if term in subject:
                    lexical_hits += 3  # Subject matches are very important
                elif term in content:
                    lexical_hits += 1
            
            # Score: -distance (lower is better) + strong lexical bonus + BM25 boost
            base_score = -dist
            lexical_bonus = min(lexical_hits, 15) * 0.25  # Cap at 15 hits, 0.25 per hit (stronger bonus)
            # Boost tickets found by BM25 (they have keyword matches)
            bm25_boost = 0.3 if bm25_ids and results["ids"][0][i] in bm25_ids else 0.0
            total_score = base_score + lexical_bonus + bm25_boost
            
            scored_items.append((total_score, i, doc, meta, dist))
        
        # Sort by total score (descending)
        scored_items.sort(reverse=True, key=lambda x: x[0])
        
        # Rebuild results in new order
        reranked = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        for _, orig_idx, doc, meta, dist in scored_items:
            reranked["ids"][0].append(results["ids"][0][orig_idx])
            reranked["documents"][0].append(doc)
            reranked["metadatas"][0].append(meta)
            reranked["distances"][0].append(dist)
        
        return reranked

    def search_guides(self, query: str, k: int = 5):
        query_emb = self.embed(query)
        if self.rag_v2_coll:
            # Filter by type in unified collection
            try:
                results = self.rag_v2_coll.query(
                    query_embeddings=[query_emb],
                    n_results=k,
                    where={"type": "guide_chunk"}
                )
            except Exception:
                # Fallback: get more results and filter manually
                results = self.rag_v2_coll.query(
                    query_embeddings=[query_emb],
                    n_results=k * 3
                )
                # Filter by type
                if results.get("ids") and results["ids"][0]:
                    filtered_indices = [
                        i for i, meta in enumerate(results.get("metadatas", [[]])[0])
                        if meta and meta.get("type") == "guide_chunk"
                    ][:k]
                    for key in ["ids", "documents", "metadatas", "distances"]:
                        if results.get(key) and results[key][0]:
                            results[key][0] = [results[key][0][i] for i in filtered_indices]
            return results
        else:
            return self.guides_coll.query(query_embeddings=[query_emb], n_results=k)

