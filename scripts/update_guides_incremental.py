import json
import pickle
import hashlib
import sys
from pathlib import Path
from tqdm import tqdm
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from config.settings import (
    PROCESSED_DIR,
    BM25_INDEX_PATH,
)
from src.phase4.vector_db import VectorDBManager

logger = logging.getLogger(__name__)


def sanitize_metadata(md: dict) -> dict:
    """Replace None values appropriately for ChromaDB compatibility."""
    clean = {}
    for k, v in md.items():
        if v is None:
            # Keep integer fields as 0, strings as empty string
            if k in ("section_index", "chunk_index"):
                clean[k] = 0
            else:
                clean[k] = ""
        else:
            clean[k] = v
    return clean


def load_chunk_file():
    path = PROCESSED_DIR / "guides_chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"Guide chunks file missing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_existing_bm25():
    if not BM25_INDEX_PATH.exists():
        return None, None

    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def rebuild_bm25_index(all_docs, all_ids):
    from rank_bm25 import BM25Okapi
    tokenized = [d.lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, all_ids


def main():
    logger.info("=== Incremental Guide Update ===")

    db = VectorDBManager()
    chunks = load_chunk_file()

    # Map new chunks
    new_map = {c["chunk_id"]: c for c in chunks}
    new_ids = set(new_map.keys())

    # Get existing Chroma IDs and documents (handle both rag_v2 and separate collections)
    existing_ids = set()
    existing_docs_map = {}
    
    if db.rag_v2_coll:
        # Use unified collection - get all guide chunk IDs
        try:
            existing_data = db.rag_v2_coll.get(where={"type": "guide_chunk"}, limit=None)
            existing_ids = set(existing_data.get("ids", []))
            existing_docs = existing_data.get("documents", [])
            existing_docs_map = {
                existing_data["ids"][i]: existing_docs[i]
                for i in range(len(existing_data["ids"]))
            }
        except Exception:
            # Fallback: get all and filter
            all_data = db.rag_v2_coll.get(limit=None)
            ids = all_data.get("ids", [])
            docs = all_data.get("documents", [])
            metas = all_data.get("metadatas", [])
            guide_indices = [i for i, meta in enumerate(metas) if meta and meta.get("type") == "guide_chunk"]
            existing_ids = {ids[i] for i in guide_indices}
            existing_docs_map = {ids[i]: docs[i] for i in guide_indices}
    elif db.guides_coll:
        existing = db.guides_coll.get()
        existing_ids = set(existing["ids"])
        existing_docs_map = {
            existing["ids"][i]: existing["documents"][i]
            for i in range(len(existing["ids"]))
        }

    to_add = []
    to_add_texts = []
    to_add_meta = []

    to_update = []
    to_update_texts = []
    to_update_meta = []

    to_delete = list(existing_ids - new_ids)

    logger.info("Checking for new/updated chunks...")

    for cid, chunk in tqdm(new_map.items()):
        text = chunk["chunk_text"].strip()
        meta = {
            "guide_number": chunk.get("guide_number") or "",
            "guide_title": chunk.get("guide_title") or "",
            "section_index": chunk.get("section_index") or 0,
            "chunk_index": chunk.get("chunk_index") or 0,
            "url": chunk.get("url") or "",
        }
        meta = sanitize_metadata(meta)

        if cid not in existing_ids:
            # NEW CHUNK
            to_add.append(cid)
            to_add_texts.append(text)
            to_add_meta.append(meta)
        else:
            # Check if modified
            old_text = existing_docs_map.get(cid, "")
            if text_hash(old_text) != text_hash(text):
                to_update.append(cid)
                to_update_texts.append(text)
                to_update_meta.append(meta)

    logger.info(f"New chunks: {len(to_add)}")
    logger.info(f"Updated chunks: {len(to_update)}")
    logger.info(f"Deleted chunks: {len(to_delete)}")

    # APPLY OPERATIONS
    # -------------------------------

    # Get collection to use
    target_coll = db.rag_v2_coll if db.rag_v2_coll else db.guides_coll
    if not target_coll:
        raise ValueError("No collection available to update guides")
    
    # Add metadata type for unified collection
    for meta in to_add_meta:
        meta["type"] = "guide_chunk"
    for meta in to_update_meta:
        meta["type"] = "guide_chunk"
    
    # Delete removed chunks
    if to_delete:
        logger.info("Deleting removed guide chunks...")
        target_coll.delete(ids=list(to_delete))

    # Add new
    if to_add:
        logger.info("Embedding new chunks...")
        emb = db.embed_batch(to_add_texts)

        target_coll.add(
            ids=to_add,
            documents=to_add_texts,
            embeddings=emb,
            metadatas=to_add_meta,
        )

    # Update changed chunks
    if to_update:
        logger.info("Embedding updated chunks...")
        emb = db.embed_batch(to_update_texts)

        # Delete old
        target_coll.delete(ids=to_update)
        # Add fresh versions
        target_coll.add(
            ids=to_update,
            documents=to_update_texts,
            embeddings=emb,
            metadatas=to_update_meta,
        )

    # REBUILD BM25 INDEX from ALL documents (tickets + guides + QA), not just guides
    logger.info("Rebuilding BM25 index (full - all types)...")
    if db.rag_v2_coll:
        all_data = db.rag_v2_coll.get(limit=None, include=["documents", "metadatas"])
        all_ids = all_data.get("ids", [])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
    elif db.guides_coll:
        gd = db.guides_coll.get()
        all_ids = gd.get("ids", [])
        all_docs = gd.get("documents", [])
        all_metas = gd.get("metadatas", [])
    else:
        all_ids, all_docs, all_metas = [], [], []

    bm25, ids = rebuild_bm25_index(all_docs, all_ids)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "docs": all_docs,
            "metadatas": all_metas,
        }, f)

    logger.info("Guide update complete!")


if __name__ == "__main__":
    main()
