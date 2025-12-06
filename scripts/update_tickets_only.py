import json
import pickle
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
    """Replace None values with empty strings for ChromaDB compatibility."""
    return {k: "" if v is None else v for k, v in md.items()}


def load_processed_tickets() -> list:
    path = PROCESSED_DIR / "processed_tickets.json"
    if not path.exists():
        raise FileNotFoundError(f"Processed tickets file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_bm25():
    if not BM25_INDEX_PATH.exists():
        return None, None

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        # Handle dictionary format from rebuild_vector_db_v2.py
        if isinstance(data, dict):
            return data.get("bm25"), data.get("ids")
        # Handle tuple format (legacy)
        elif isinstance(data, tuple):
            return data[0], data[1]
        else:
            return None, None


def update_bm25_index(existing_bm25, existing_ids, new_docs, new_ids):
    from rank_bm25 import BM25Okapi

    if existing_bm25 is None:
        # build fresh index
        tokenized = [doc.lower().split() for doc in new_docs]
        bm25 = BM25Okapi(tokenized)
        return bm25, new_ids

    # append incremental docs
    all_docs = existing_ids + new_ids
    all_text = existing_bm25.corpus + [doc.lower().split() for doc in new_docs]

    bm25 = BM25Okapi(all_text)
    return bm25, all_docs


def main():
    logger.info("=== Incremental Ticket Update ===")

    db = VectorDBManager()

    # Load full processed tickets
    tickets = load_processed_tickets()

    # Load existing IDs from Chroma
    existing_ids = set(db.tickets_coll.get()["ids"])

    new_ids = []
    new_docs = []
    new_meta = []

    for t in tickets:
        tid = t.get("ticket_id")
        if not tid:
            continue

        ticket_uid = f"ticket_{tid}"
        if ticket_uid in existing_ids:
            continue  # already indexed → skip

        text = t.get("searchable_text", "").strip()
        if not text:
            continue

        new_ids.append(ticket_uid)
        new_docs.append(text)

        meta = {
            "ticket_id": tid,
            "subject": t.get("subject") or "",
            "status": t.get("status") or "",
            "priority": t.get("priority") or "",
            "created_at": t.get("created_at") or "",
        }
        new_meta.append(sanitize_metadata(meta))

    if not new_ids:
        logger.info("No new tickets to index. Exiting.")
        return

    logger.info(f"Found {len(new_ids)} new tickets → embedding...")

    embeddings = db.embed_batch(new_docs)

    logger.info("Adding to Chroma...")
    db.tickets_coll.add(
        ids=new_ids,
        documents=new_docs,
        embeddings=embeddings,
        metadatas=new_meta,
    )

    logger.info("Updating BM25 index...")

    existing_bm25, existing_bm25_ids = load_existing_bm25()
    bm25, final_ids = update_bm25_index(existing_bm25, existing_bm25_ids or [], new_docs, new_ids)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, final_ids), f)

    logger.info(f"Completed: Added {len(new_ids)} new tickets.")


if __name__ == "__main__":
    main()
