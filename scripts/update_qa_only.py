#!/usr/bin/env python3
"""
Extract QA pairs from processed tickets and create embeddings.
Combines extraction + embedding in one step with progress display.
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

# Make repo root importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Config import
from config.settings import (
    PROCESSED_DIR,
    CHROMA_DB_DIR,
    LOCAL_EMBEDDING_MODEL,
    BM25_INDEX_PATH,
)

# LangChain imports (with fallback)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("update_qa_only")


# ----------------------------
# QA Extraction Logic
# ----------------------------
def extract_qa_pairs_from_ticket(ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract QA pairs from a single ticket."""
    qa_pairs = []
    ticket_id = ticket.get("ticket_id")
    subject = ticket.get("subject", "")
    
    conversation = ticket.get("conversation", [])
    if not conversation:
        return []
    
    agent_id = ticket.get("agent_id")
    used_agent_indices = set()
    
    for i, comment in enumerate(conversation):
        author_id = comment.get("author_id")
        
        if author_id == agent_id or author_id == -1:
            continue
        
        customer_msg = comment.get("body") or comment.get("plain_body") or ""
        customer_msg = customer_msg.strip()
        
        if len(customer_msg) < 20:
            continue
        
        agent_response = None
        agent_idx = None
        for j in range(i + 1, len(conversation)):
            next_comment = conversation[j]
            if next_comment.get("author_id") == agent_id:
                if j not in used_agent_indices:
                    agent_response = next_comment.get("body") or next_comment.get("plain_body") or ""
                    agent_response = agent_response.strip()
                    agent_idx = j
                    break
        
        if not agent_response or len(agent_response) < 20:
            continue
        
        if agent_idx:
            used_agent_indices.add(agent_idx)
        
        qa_id = f"{ticket_id}_qa_{len(qa_pairs)}"
        qa_pair = {
            "qa_id": qa_id,
            "orig_ticket_id": ticket_id,
            "subject": subject,
            "customer_question": customer_msg,
            "agent_answer": agent_response,
            "full_text": f"DOMANDA: {customer_msg}\n\nRISPOSTA: {agent_response}",
            "type": "qa_pair",
            "extraction_method": "conversation_flow"
        }
        qa_pairs.append(qa_pair)
    
    return qa_pairs


# ----------------------------
# BM25 helpers
# ----------------------------
def load_bm25(path: Path):
    if not path.exists():
        logger.info("No BM25 index found — creating new.")
        return None, [], [], []
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data["bm25"], data["ids"], data["docs"], data.get("metadatas", [])


def save_bm25(path: Path, bm25, ids, docs, metadatas=None):
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metadatas": metadatas or []}, f)
    logger.info(f"Saved BM25 index ({len(ids)} docs, {len(metadatas or [])} metadatas).")


# ----------------------------
# Main script
# ----------------------------
def main():
    logger.info("=" * 60)
    logger.info("QA PAIR EXTRACTION + EMBEDDING")
    logger.info("=" * 60)
    
    # Step 1: Load processed tickets
    processed_file = PROCESSED_DIR / "processed_tickets.json"
    if not processed_file.exists():
        logger.error(f"Processed tickets not found: {processed_file}")
        logger.info("Please upload and process tickets first.")
        return 1
    
    with open(processed_file, "r", encoding="utf-8") as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets)} processed tickets")
    
    # Step 2: Extract QA pairs
    logger.info("\n📝 Extracting QA pairs from conversations...")
    all_qa_pairs = []
    
    for ticket in tqdm(tickets, desc="Extracting QA pairs", unit="ticket"):
        try:
            pairs = extract_qa_pairs_from_ticket(ticket)
            all_qa_pairs.extend(pairs)
        except Exception as e:
            continue
    
    logger.info(f"✅ Extracted {len(all_qa_pairs)} QA pairs")
    
    if not all_qa_pairs:
        logger.warning("No QA pairs extracted. Nothing to embed.")
        return 0
    
    # Save QA pairs to file
    qa_file = PROCESSED_DIR / "qa_pairs.json"
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved QA pairs to {qa_file}")
    
    # Step 3: Initialize embedding model
    logger.info(f"\n🔧 Loading embedding model: {LOCAL_EMBEDDING_MODEL}")
    embedder = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    logger.info("✅ Embedding model loaded!")
    
    # Step 4: Initialize Chroma
    chroma = Chroma(
        collection_name="rag_v2",
        embedding_function=embedder,
        persist_directory=str(CHROMA_DB_DIR),
    )
    
    col = chroma._collection
    existing = col.get(include=[])
    existing_ids = set(existing.get("ids", []))
    logger.info(f"ChromaDB has {len(existing_ids)} existing documents")
    
    # Step 5: Prepare new QA pairs for embedding
    new_texts = []
    new_ids = []
    new_metas = []
    
    for qa in all_qa_pairs:
        qa_id = qa.get("qa_id")
        if qa_id in existing_ids:
            continue
        
        text = qa.get("full_text", "")
        if not text.strip():
            continue
        
        new_texts.append(text.strip())
        new_ids.append(qa_id)
        new_metas.append({
            "type": "qa_pair",
            "qa_id": qa_id,
            "orig_ticket_id": qa.get("orig_ticket_id"),
            "subject": qa.get("subject", ""),
            "extraction_method": qa.get("extraction_method", ""),
        })
    
    if not new_texts:
        logger.info("No new QA pairs to embed — all already in ChromaDB.")
        return 0
    
    logger.info(f"\n🆕 New QA pairs to embed: {len(new_texts)}")
    
    # Step 6: Generate embeddings with progress
    logger.info("🔧 Generating embeddings...")
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(new_texts), batch_size), desc="Embedding QA pairs"):
        batch = new_texts[i:i + batch_size]
        batch_emb = embedder.embed_documents(batch)
        embeddings.extend(batch_emb)
    
    logger.info("✅ Embeddings complete!")
    
    # Step 7: Add to ChromaDB
    logger.info("\n📦 Adding QA pairs to ChromaDB...")
    CHROMA_BATCH_SIZE = 5000
    
    try:
        total_added = 0
        for i in range(0, len(new_ids), CHROMA_BATCH_SIZE):
            batch_end = min(i + CHROMA_BATCH_SIZE, len(new_ids))
            batch_ids = new_ids[i:batch_end]
            batch_docs = new_texts[i:batch_end]
            batch_embs = embeddings[i:batch_end]
            batch_metas = new_metas[i:batch_end]
            
            col.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=batch_metas,
            )
            total_added += len(batch_ids)
            logger.info(f"Added batch {i//CHROMA_BATCH_SIZE + 1}: {total_added}/{len(new_ids)} QA pairs")
        
        logger.info("✅ ChromaDB updated!")
    except Exception as e:
        logger.exception(f"ChromaDB insertion failed: {e}")
        return 1
    
    # Step 8: Update BM25 index
    logger.info("\n📊 Updating BM25 index...")
    bm25, bm25_ids, bm25_docs, bm25_metas = load_bm25(Path(BM25_INDEX_PATH))
    
    if bm25 is None:
        bm25_docs = new_texts.copy()
        bm25_ids = new_ids.copy()
        bm25_metas = new_metas.copy()
    else:
        bm25_docs.extend(new_texts)
        bm25_ids.extend(new_ids)
        bm25_metas.extend(new_metas)
    
    bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])
    save_bm25(Path(BM25_INDEX_PATH), bm25, bm25_ids, bm25_docs, bm25_metas)
    
    # Done!
    logger.info("\n" + "=" * 60)
    logger.info("🎉 QA EXTRACTION + EMBEDDING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"  • Extracted: {len(all_qa_pairs)} QA pairs")
    logger.info(f"  • Embedded: {len(new_texts)} new QA pairs")
    logger.info(f"  • ChromaDB + BM25 updated")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
