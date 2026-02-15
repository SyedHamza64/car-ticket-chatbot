#!/usr/bin/env python3
"""
Incremental ticket updater (LangChain + HuggingFaceEmbeddings + Chroma + BM25)
Uses same ID format and chunking as rebuild_vector_db_v2.py:
  - IDs: ticket_{id}__idx{chunk}  (e.g. ticket_5122__idx0)
  - Metadata: orig_ticket_id, type, subject, status, priority, created_at
  - Long tickets are chunked at 9000 chars
"""

import sys
import re
import json
import pickle
import logging
from pathlib import Path

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
# Text cleaning & chunking (same as rebuild_vector_db_v2.py)
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:12000]


def maybe_chunk(text: str, max_chars=9000):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max_chars - 200
    for i in range(0, len(text), step):
        chunks.append(text[i:i + max_chars])
    return chunks


def sanitize_metadata(md: dict) -> dict:
    return {k: ("" if v is None else v) for k, v in md.items()}


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
    logger.info("=== Incremental Ticket Updater ===")

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

    # Detect new tickets -- use same format as rebuild_vector_db_v2.py
    new_texts = []
    new_ids = []
    new_metas = []

    for t in tqdm(tickets, desc="Preparing tickets", unit="ticket"):
        tid = t.get("ticket_id")
        if tid is None:
            continue

        # Check if ANY chunk of this ticket already exists
        base_id = f"ticket_{tid}__idx0"
        if base_id in existing_ids:
            continue

        raw = t.get("searchable_text", "")
        cleaned = clean_text(raw)
        if not cleaned:
            continue

        parts = maybe_chunk(cleaned)
        for idx, chunk in enumerate(parts):
            uid = f"ticket_{tid}__idx{idx}"
            if uid in existing_ids:
                continue

            new_texts.append(chunk)
            new_ids.append(uid)
            new_metas.append(sanitize_metadata({
                "type": "ticket",
                "orig_ticket_id": tid,
                "ticket_id": tid,
                "chunk_index": idx,
                "subject": t.get("subject", ""),
                "status": t.get("status", ""),
                "priority": t.get("priority", ""),
                "created_at": t.get("created_at", ""),
            }))

    if not new_texts:
        logger.info("No new tickets to embed — everything is up to date.")
        return

    logger.info(f"New ticket chunks to embed: {len(new_texts)}")
    logger.info("Generating embeddings...")

    # Embed with progress
    embeddings = []
    batch_size = 32

    for i in tqdm(range(0, len(new_texts), batch_size), desc="Embedding tickets"):
        batch = new_texts[i: i + batch_size]
        batch_emb = embedder.embed_documents(batch)
        embeddings.extend(batch_emb)

    logger.info("Embeddings complete.")

    # Add to Chroma (in batches)
    logger.info("Adding tickets to Chroma...")
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
            logger.info(f"Added batch {i//CHROMA_BATCH_SIZE + 1}: {total_added}/{len(new_ids)} ticket chunks")

        logger.info("Chroma updated successfully.")
    except Exception as e:
        logger.exception(f"Chroma insertion failed: {e}")
        return

    # BM25 update
    bm25, bm25_ids, bm25_docs, bm25_metas = load_bm25(Path(BM25_INDEX_PATH))

    if bm25 is None:
        bm25_docs = new_texts.copy()
        bm25_ids = new_ids.copy()
        bm25_metas = new_metas.copy()
        bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])
    else:
        bm25_docs.extend(new_texts)
        bm25_ids.extend(new_ids)
        bm25_metas.extend(new_metas)
        bm25 = BM25Okapi([d.lower().split() for d in bm25_docs])

    save_bm25(Path(BM25_INDEX_PATH), bm25, bm25_ids, bm25_docs, bm25_metas)

    logger.info("Update complete — embedding + Chroma + BM25 updated.")


if __name__ == "__main__":
    main()
