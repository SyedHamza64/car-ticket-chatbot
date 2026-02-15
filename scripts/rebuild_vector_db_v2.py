"""
FINAL – LOCAL EMBEDDINGS VERSION
RAG v2 Vector DB Rebuild Script
--------------------------------
- Uses SentenceTransformer (free, local)
- Determines unique IDs (no duplicates)
- Cleans + conditionally chunks text
- Sanitizes metadata (no None → crash fix)
- Rebuilds ChromaDB collection
- Builds BM25 index
"""

import os
import sys
import re
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from dotenv import load_dotenv

import chromadb
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------
# FIX IMPORT PATH + LOAD .env
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import (
    PROCESSED_TICKETS_FILE,
    GUIDES_CHUNKS_FILE,
    CHROMA_DB_DIR,
    BM25_INDEX_PATH,
    LOCAL_EMBEDDING_MODEL,
)

# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    # Hard limit for safety
    MAX_CHARS = 12000
    return text[:MAX_CHARS]


# ---------------------------------------------------------
# CONDITIONAL CHUNKING
# ---------------------------------------------------------
def maybe_chunk(text: str, max_chars=9000):
    """Chunk only when needed."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max_chars - 200
    for i in range(0, len(text), step):
        chunks.append(text[i:i + max_chars])
    return chunks


# ---------------------------------------------------------
# SANITIZE METADATA (NO None ALLOWED IN CHROMA)
# ---------------------------------------------------------
def sanitize_metadata(md: dict) -> dict:
    clean = {}
    for k, v in md.items():
        clean[k] = "" if v is None else v
    return clean


# ---------------------------------------------------------
# LANGCHAIN EMBEDDING GENERATOR
# ---------------------------------------------------------
def get_embeddings(model_name=None):
    """Get LangChain HuggingFaceEmbeddings instance."""
    model_name = model_name or LOCAL_EMBEDDING_MODEL
    print(f"\n🔧 Loading LangChain embeddings: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# QA PAIR EXTRACTION (from tickets)
# ---------------------------------------------------------
def extract_qa_pairs_from_ticket(ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract QA pairs from a single ticket conversation."""
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

        customer_msg = (comment.get("body") or comment.get("plain_body") or "").strip()
        if len(customer_msg) < 20:
            continue

        agent_response = None
        agent_idx = None
        for j in range(i + 1, len(conversation)):
            next_comment = conversation[j]
            if next_comment.get("author_id") == agent_id:
                if j not in used_agent_indices:
                    agent_response = (next_comment.get("body") or next_comment.get("plain_body") or "").strip()
                    agent_idx = j
                    break

        if not agent_response or len(agent_response) < 20:
            continue
        if agent_idx is not None:
            used_agent_indices.add(agent_idx)

        qa_id = f"{ticket_id}_qa_{len(qa_pairs)}"
        qa_pairs.append({
            "qa_id": qa_id,
            "orig_ticket_id": ticket_id,
            "subject": subject,
            "full_text": f"DOMANDA: {customer_msg}\n\nRISPOSTA: {agent_response}",
            "type": "qa_pair",
        })
    return qa_pairs


# ---------------------------------------------------------
# BUILD BM25
# ---------------------------------------------------------
def build_bm25(docs):
    tokenized = [d.split() for d in docs]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------
# MAIN REBUILD PROCESS
# ---------------------------------------------------------
def main():
    print("\n=== 🚀 FULL VECTOR DB REBUILD (LOCAL EMBEDDINGS) ===\n")

    # -----------------------------
    # Load Tickets
    # -----------------------------
    tickets = []
    if PROCESSED_TICKETS_FILE.exists():
        tickets = load_json(Path(PROCESSED_TICKETS_FILE))
        print(f"Loaded {len(tickets)} tickets.")
    else:
        print("⚠ No processed tickets found — skipping.")

    # -----------------------------
    # Load Guide Chunks
    # -----------------------------
    guide_chunks = load_json(Path(GUIDES_CHUNKS_FILE))
    print(f"Loaded {len(guide_chunks)} guide chunks.")

    # Prepare all containers
    bm25_docs = []
    bm25_ids = []
    dense_docs = []
    dense_ids = []
    dense_meta = []

    # -----------------------------
    # Process Tickets (with progress bar)
    # -----------------------------
    print("\n📋 Processing tickets...")
    for t in tqdm(tickets, desc="Processing tickets", unit="ticket"):
        raw = t.get("searchable_text", "")
        cleaned = clean_text(raw)
        if not cleaned:
            continue

        parts = maybe_chunk(cleaned)

        for idx, chunk in enumerate(parts):
            uid = f"ticket_{t['ticket_id']}__idx{idx}"

            bm25_docs.append(chunk)
            bm25_ids.append(uid)

            dense_docs.append(chunk)
            dense_ids.append(uid)

            meta = {
                "type": "ticket",
                "orig_ticket_id": t.get("ticket_id"),
                "ticket_id": t.get("ticket_id"),  # Add ticket_id for filtering
                "chunk_index": idx,
                "subject": t.get("subject", ""),
                "status": t.get("status", ""),
                "priority": t.get("priority", ""),
                "created_at": t.get("created_at", ""),
            }

            dense_meta.append(sanitize_metadata(meta))

    # -----------------------------
    # Process Guide Chunks (with progress bar)
    # -----------------------------
    print("\n📚 Processing guide chunks...")
    for gc in tqdm(guide_chunks, desc="Processing guides", unit="chunk"):
        cleaned = clean_text(gc.get("chunk_text", ""))
        if not cleaned:
            continue

        cid = gc.get("chunk_id")
        if not cid:
            cid = f"{gc.get('guide_number','GUIDE')}__sec{gc.get('section_index',0)}__chunk{gc.get('chunk_index',0)}"

        bm25_docs.append(cleaned)
        bm25_ids.append(cid)

        dense_docs.append(cleaned)
        dense_ids.append(cid)

        meta = {
            "type": "guide_chunk",
            "guide_number": gc.get("guide_number"),
            "guide_title": gc.get("guide_title"),
            "section_index": gc.get("section_index"),
            "chunk_index": gc.get("chunk_index"),
            "url": gc.get("url"),
        }
        dense_meta.append(sanitize_metadata(meta))

    # -----------------------------
    # Process QA pairs (from tickets)
    # -----------------------------
    print("\n❓ Extracting QA pairs from tickets...")
    all_qa_pairs = []
    for t in tqdm(tickets, desc="Extracting QA pairs", unit="ticket"):
        try:
            pairs = extract_qa_pairs_from_ticket(t)
            all_qa_pairs.extend(pairs)
        except Exception:
            continue

    for qa in all_qa_pairs:
        text = clean_text(qa.get("full_text", ""))
        if not text:
            continue
        qa_id = qa.get("qa_id", "")
        if not qa_id:
            continue

        bm25_docs.append(text)
        bm25_ids.append(qa_id)
        dense_docs.append(text)
        dense_ids.append(qa_id)
        meta = {
            "type": "qa_pair",
            "qa_id": qa_id,
            "orig_ticket_id": qa.get("orig_ticket_id"),
            "subject": qa.get("subject", ""),
        }
        dense_meta.append(sanitize_metadata(meta))

    print(f"   Added {len(all_qa_pairs)} QA pairs.")
    print(f"\n✅ Total documents prepared for embedding: {len(dense_docs)}", flush=True)

    # -----------------------------
    # Embeddings using LangChain
    # -----------------------------
    print("\n🔧 Loading embedding model...", flush=True)
    embeddings = get_embeddings()
    print("✅ Embedding model loaded!", flush=True)
    
    # Convert to LangChain Documents
    print("\n📝 Creating document objects...", flush=True)
    langchain_docs = []
    for i, (doc_text, doc_id, doc_meta) in enumerate(zip(dense_docs, dense_ids, dense_meta)):
        langchain_docs.append(Document(
            page_content=doc_text,
            metadata={**doc_meta, "id": doc_id}
        ))
    print(f"✅ Created {len(langchain_docs)} document objects", flush=True)
    
    # Duplicate ID check
    if len(dense_ids) != len(set(dense_ids)):
        raise SystemExit("❌ Duplicate IDs detected. Aborting.")

    # -----------------------------
    # Chroma DB using LangChain — Drop + Recreate (with batched progress)
    # -----------------------------
    print("\n📦 Deleting old ChromaDB collection...", flush=True)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        client.delete_collection("rag_v2")
        print("✅ Old collection deleted.", flush=True)
    except Exception as e:
        print(f"  (No existing collection to delete: {e})", flush=True)

    # Batch insert with progress
    BATCH_SIZE = 500
    total_batches = (len(langchain_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🚀 Generating embeddings & inserting to ChromaDB...", flush=True)
    print(f"   Total: {len(langchain_docs)} docs in {total_batches} batches ({BATCH_SIZE} docs/batch)", flush=True)
    print(f"   Estimated time: 5-10 minutes on CPU\n", flush=True)
    
    vectorstore = None
    start_time = time.time()
    
    for batch_num, batch_idx in enumerate(range(0, len(langchain_docs), BATCH_SIZE), 1):
        batch_docs = langchain_docs[batch_idx:batch_idx + BATCH_SIZE]
        batch_ids = dense_ids[batch_idx:batch_idx + BATCH_SIZE]
        batch_start = time.time()
        
        print(f"   📦 Batch {batch_num}/{total_batches}: Processing {len(batch_docs)} docs...", end=" ", flush=True)
        
        if vectorstore is None:
            # First batch: create the vectorstore with explicit ids so get(ids=[...]) works
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name="rag_v2",
                persist_directory=str(CHROMA_DB_DIR),
                ids=batch_ids,
            )
        else:
            # Subsequent batches: add with same explicit ids
            vectorstore.add_documents(batch_docs, ids=batch_ids)
        
        batch_time = time.time() - batch_start
        elapsed_total = time.time() - start_time
        remaining_batches = total_batches - batch_num
        eta = (elapsed_total / batch_num) * remaining_batches
        
        print(f"Done! ({batch_time:.1f}s) | Elapsed: {elapsed_total:.0f}s | ETA: {eta:.0f}s", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Chroma indexing complete! (took {elapsed:.1f} seconds)", flush=True)

    # -----------------------------
    # BM25 INDEX
    # -----------------------------
    print("\n📚 Building BM25 Index...", flush=True)
    bm25 = build_bm25(bm25_docs)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": bm25_ids,
            "docs": bm25_docs,
            "metadatas": dense_meta,
        }, f)

    print(f"✅ BM25 saved at: {BM25_INDEX_PATH}")
    print("\n🎉 REBUILD COMPLETE — Local embeddings are now active!\n")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
