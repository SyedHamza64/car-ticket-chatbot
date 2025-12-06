# config/settings.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GUIDES_DIR = DATA_DIR / "guides"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

PROCESSED_TICKETS_FILE = PROCESSED_DIR / "processed_tickets.json"
GUIDES_COMBINED_FILE = GUIDES_DIR / "guides.json"
GUIDES_CHUNKS_FILE = PROCESSED_DIR / "guides_chunks.json"

# -----------------------------
# Embedding config (LOCAL ONLY)
# -----------------------------
EMBEDDING_PROVIDER = "local"

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)


# BM25 config
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"

# Chroma
CHROMA_DB_DIR = CHROMA_DIR

# Chunker settings
CHUNK_TARGET_WORDS = int(os.getenv("CHUNK_TARGET_WORDS", 250))  # roughly 150-350 tokens target
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 30))

# Hybrid retrieval defaults (will be used later)
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", 0.65))
HYBRID_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", 0.35))

# Reranker model (pluggable)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Groq API key (for answer generation)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Gemini API key (for answer generation)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Ollama settings (for local LLM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

# Misc
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
