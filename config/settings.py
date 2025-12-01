"""Configuration settings for the RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GUIDES_DATA_DIR = DATA_DIR / "guides"
LOGS_DIR = BASE_DIR / "logs"
CHROMA_DATA_DIR = DATA_DIR / "chroma"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, GUIDES_DATA_DIR, LOGS_DIR, CHROMA_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Ollama LLM settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# Groq API settings (Fast LLM Inference)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()  # Strip whitespace
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # Default Groq model
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Embedding model (upgraded to all-mpnet-base-v2 for better retrieval quality)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", "50"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE_NAME = os.getenv("LOG_FILE", "app.log")
LOG_FILE = LOGS_DIR / LOG_FILE_NAME if not os.path.isabs(LOG_FILE_NAME) else Path(LOG_FILE_NAME)

# Zendesk export file (combined export)
ZENDESK_EXPORT_FILE = RAW_DATA_DIR / "export_combined.json"

