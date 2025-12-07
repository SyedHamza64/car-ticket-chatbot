# LaCuraDellAuto AI Support Assistant

AI-powered support assistant for LaCuraDellAuto customer service team using RAG (Retrieval-Augmented Generation) architecture.

## Overview

This system helps support agents provide accurate, on-brand customer responses by:
- Learning from historical Zendesk support tickets
- Referencing official LCDA technical guides
- Maintaining brand tone and consistency
- Extracting product recommendations with links
- Providing step-by-step procedures

## Features

- **Fast Response Time**: 2-3 seconds (fast mode) or 10-12 seconds (full quality mode)
- **Multiple LLM Providers**: Support for Groq, Gemini, and Ollama
- **Hybrid Search**: Combines semantic (dense) and keyword (sparse) retrieval
- **Product Link Extraction**: Automatically extracts and includes product URLs
- **Incremental Updates**: Fast updates for new tickets and guides
- **Modern UI**: Clean Streamlit interface with dark/light theme

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# LLM Provider (choose at least one)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OR
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash

# OR (for local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct

# Embedding Model (local, no API key needed)
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Optional: Enable GPU if available
USE_CUDA=false

# Logging
LOG_LEVEL=INFO
```

### 3. Prepare Data Files

Ensure these files exist:
- `data/processed/processed_tickets.json` - Processed support tickets
- `data/guides/guides.json` - Scraped product guides  
- `data/chroma/` - ChromaDB vector database (populated)
- `data/bm25_index.pkl` - BM25 sparse index

**To set up data from scratch:**
1. Process tickets: `python -m src.phase2.process_tickets`
2. Scrape guides: `python -m src.phase3.scrape_guides_fast`
3. Chunk guides: `python -m src.phase1.semantic_chunker`
4. Build database: `python scripts/rebuild_vector_db_v2.py`

### 4. Launch Application

```bash
streamlit run streamlit_app.py --server.port 8501
```

Access at: `http://localhost:8501`

## Project Structure

```
chat-bot-ticket/
├── streamlit_app.py              # Main web interface
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (not in git)
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── DEPLOYMENT.md                # Production deployment guide
├── config/
│   └── settings.py              # Configuration settings
├── src/
│   ├── phase1/
│   │   └── semantic_chunker.py  # Guide chunking
│   ├── phase2/
│   │   └── process_tickets.py   # Ticket processing
│   ├── phase3/
│   │   └── scrape_guides_fast.py # Guide scraping
│   ├── phase4/
│   │   ├── rag_pipeline_langchain.py  # Main RAG pipeline
│   │   └── vector_db.py         # Vector database manager
│   └── utils/
│       ├── logger.py            # Logging utilities
│       └── model_checker.py     # Model validation
├── scripts/
│   ├── rebuild_vector_db_v2.py  # Full database rebuild
│   ├── update_tickets_only.py   # Incremental ticket updates
│   ├── update_guides_incremental.py  # Incremental guide updates
│   └── import_new_tickets.py    # Import new tickets
└── data/                        # Data files (excluded from git)
    ├── processed/               # Processed tickets
    ├── guides/                  # Scraped guides
    ├── chroma/                  # ChromaDB vector database
    └── bm25_index.pkl          # BM25 sparse index
```

## Usage

### Adding New Tickets

1. Upload NDJSON file via Streamlit interface (Manage Knowledge Base tab)
2. Click "🚀 Process & Update Knowledge Base"
3. System automatically processes and indexes new tickets

### Updating Guides

1. Click "🔄 Refresh Guides" in Streamlit
2. System scrapes latest guides from website
3. Automatically chunks and indexes new content

### Querying

1. Enter customer question in the main interface
2. Select number of tickets/guides to retrieve
3. Choose AI provider (Groq/Gemini/Ollama)
4. Click "✨ Generate Response"
5. Review answer with sources

## Configuration

### LLM Providers

**Groq (Recommended - Fast)**
- Fast inference (1-2 seconds)
- Free tier available
- Model: `llama-3.3-70b-versatile`

**Gemini (Google)**
- Good quality
- Free tier available
- Model: `models/gemini-2.5-flash`

**Ollama (Local)**
- No API costs
- Requires local setup
- Slower but private

### Performance Modes

**Fast Mode (Default)**
- Response time: 2-3 seconds
- Dense retrieval only
- Good quality for most queries

**Full Mode**
- Response time: 10-12 seconds
- Hybrid search + reranking
- Best quality for complex queries

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq API / Gemini API / Ollama (local)
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers (HuggingFace)
- **Sparse Search**: BM25
- **Reranking**: CrossEncoder (optional)
- **Language**: Python 3.8+

## Troubleshooting

### API Errors
- Verify API keys in `.env` file
- Check API provider status
- Ensure sufficient API quota

### Slow Performance
- Enable fast_mode (default)
- Reduce number of retrieved documents
- Check internet connection for API calls

### Database Issues
- Verify ChromaDB files exist in `data/chroma/`
- Check BM25 index exists: `data/bm25_index.pkl`
- Rebuild database if corrupted: Use "Full Rebuild" button

### CUDA Out of Memory
- Set `USE_CUDA=false` in `.env`
- System will use CPU (slower but stable)

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed production deployment instructions including:
- Docker setup
- VPS deployment
- Streamlit Cloud
- Systemd service configuration
- Nginx reverse proxy

## License

Internal use only - LaCuraDellAuto
