#!/usr/bin/env python3
"""
Incremental ticket updater (LangChain + HuggingFaceEmbeddings + Chroma + BM25)
⭐ Includes a visible tqdm progress bar for embeddings ⭐
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

# Make repo root importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Config import
from config.settings import (
    PROCESSED_TICKETS_FILE,
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
logger = logging.getLogger("update_tickets_only")


# ----------------------------
# Load & save helpers
# ----------------------------
def load_processed_tickets(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} processed tickets.")
    return data


def load_bm25(path: Path):
    if not path.exists():
        logger.info("No BM25 index found — creating new.")
        return None, [], []

    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["ids"], data["docs"]


def save_bm25(path: Path, bm25, ids, docs):
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs}, f)
    logger.info(f"Saved BM25 index ({len(ids)} docs).")


# ----------------------------
# Main script
# ----------------------------
def main():
    logger.info("=== Incremental Ticket Updater (with tqdm progress) ===")

    # Load tickets
    tickets = load_processed_tickets(Path(PROCESSED_TICKETS_FILE))

    # Init embeddings
    logger.info(f"Loading embedding model: {LOCAL_EMBEDDING_MODEL}")
    embedder = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    # Init Chroma
    chroma = Chroma(
        collection_name="rag_v2",
        embedding_function=embedder,
        persist_directory=str(CHROMA_DB_DIR),
    )

    # Fetch existing IDs
    col = chroma._collection
    existing = col.get(include=[])
    existing_ids = set(existing.get("ids", []))
    logger.info(f"Chroma has {len(existing_ids)} existing documents.")

    # Detect new tickets
    new_texts = []
    new_ids = []
    new_metas = []

    for t in tickets:
        tid = t.get("ticket_id")
        if tid is None:
            continue
        uid = f"ticket_{tid}"

        if uid not in existing_ids:
            text = t.get("searchable_text") or t.get("description") or ""
            if not text.strip():
                continue

            new_texts.append(text.strip())
            new_ids.append(uid)
            new_metas.append({
                "type": "ticket",
                "ticket_id": tid,
                "subject": t.get("subject", ""),
                "status": t.get("status", ""),
            })

    if not new_texts:
        logger.info("No new tickets to embed — everything is up to date.")
        return

    logger.info(f"🆕 New tickets detected: {len(new_texts)}")
    logger.info("🔧 Generating embeddings with tqdm progress bar...")

    # ----------------------------
    # ⭐ MANUAL EMBEDDING WITH TQDM ⭐
    # ----------------------------
    embeddings = []
    batch_size = 32

    for i in tqdm(range(0, len(new_texts), batch_size), desc="Embedding tickets"):
        batch = new_texts[i: i + batch_size]
        batch_emb = embedder.embed_documents(batch)
        embeddings.extend(batch_emb)

    logger.info("Embeddings complete.")

    # ----------------------------
    # Add to Chroma (in batches to avoid size limit)
    # ----------------------------
    logger.info("📦 Adding tickets to Chroma...")
    CHROMA_BATCH_SIZE = 5000  # ChromaDB max is 5461, use 5000 for safety
    
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
            logger.info(f"Added batch {i//CHROMA_BATCH_SIZE + 1}: {total_added}/{len(new_ids)} tickets")
        
        logger.info("Chroma updated successfully.")
    except Exception as e:
        logger.exception(f"Chroma insertion failed: {e}")
        return

    # ----------------------------
    # BM25 update
    # ----------------------------
    bm25, bm25_ids, bm25_docs = load_bm25(Path(BM25_INDEX_PATH))

    if bm25 is None:
        bm25_docs = new_texts.copy()
        bm25_ids = new_ids.copy()
        bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])
    else:
        bm25_docs.extend(new_texts)
        bm25_ids.extend(new_ids)
        bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])

    save_bm25(Path(BM25_INDEX_PATH), bm25, bm25_ids, bm25_docs)

    logger.info("🎉 Update complete — embedding + Chroma + BM25 updated.")


if __name__ == "__main__":
    main()
