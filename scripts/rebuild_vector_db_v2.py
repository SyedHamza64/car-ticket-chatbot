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
    # Process Tickets
    # -----------------------------
    for t in tickets:
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
                "chunk_index": idx,
                "subject": t.get("subject", ""),
                "status": t.get("status", ""),
                "priority": t.get("priority", ""),
                "created_at": t.get("created_at", ""),
            }

            dense_meta.append(sanitize_metadata(meta))

    # -----------------------------
    # Process Guide Chunks
    # -----------------------------
    for gc in guide_chunks:
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

    print(f"\nTotal documents prepared for embedding: {len(dense_docs)}")

    # -----------------------------
    # Embeddings using LangChain
    # -----------------------------
    print("\n🔧 Generating embeddings using LangChain…")
    embeddings = get_embeddings()
    
    # Convert to LangChain Documents
    langchain_docs = []
    for i, (doc_text, doc_id, doc_meta) in enumerate(zip(dense_docs, dense_ids, dense_meta)):
        langchain_docs.append(Document(
            page_content=doc_text,
            metadata={**doc_meta, "id": doc_id}
        ))
    
    # Duplicate ID check
    if len(dense_ids) != len(set(dense_ids)):
        raise SystemExit("❌ Duplicate IDs detected. Aborting.")

    # -----------------------------
    # Chroma DB using LangChain — Drop + Recreate
    # -----------------------------
    print("📦 Writing to ChromaDB using LangChain…")

    # Delete existing collection if it exists
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        client.delete_collection("rag_v2")
    except:
        pass

    # Use LangChain's Chroma.from_documents
    vectorstore = Chroma.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        collection_name="rag_v2",
        persist_directory=str(CHROMA_DB_DIR),
    )

    print("✓ Chroma indexing complete using LangChain.\n")

    # -----------------------------
    # BM25 INDEX
    # -----------------------------
    print("📚 Building BM25 Index…")

    bm25 = build_bm25(bm25_docs)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": bm25_ids,
            "docs": bm25_docs
        }, f)

    print(f"✓ BM25 saved at: {BM25_INDEX_PATH}")
    print("\n🎉 REBUILD COMPLETE — Local embeddings are now active!\n")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
