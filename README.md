# LaCuraDellAuto AI Support Assistant

AI-powered support assistant for LaCuraDellAuto customer service team using RAG (Retrieval-Augmented Generation) architecture.

## Overview

This system helps support agents draft accurate, on-brand customer responses by:
- Learning from 2,590 historical Zendesk tickets
- Referencing 15 official LCDA technical guides (83 sections)
- Maintaining brand tone and consistency
- Generating multiple draft variations for operator selection

## Project Structure (Runtime Branch)

```
chat-bot-ticket/
├── streamlit_app.py       # Main web interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── .gitignore            # Git ignore rules
├── config/
│   └── settings.py        # Configuration settings
├── src/
│   ├── phase4/
│   │   ├── vector_db.py            # ChromaDB vector database
│   │   └── rag_pipeline.py         # RAG orchestration
│   └── utils/
│       ├── logger.py               # Logging utilities
│       └── model_checker.py        # Model validation
├── scripts/
│   └── run_streamlit.py            # Streamlit launcher
└── data/                           # Data files (excluded from git)
    ├── processed/                  # Processed tickets (required)
    ├── guides/                     # Scraped guides (required)
    └── chroma/                     # Vector database (required)
```

> **For setup tools and development:** See the `setup` branch which includes:
> - Data processing scripts (`src/phase2/`, `src/phase3/`)
> - Database setup (`src/phase4/populate_vector_db.py`)
> - Tests (`tests/`)
> - Diagnostics (`diagnostics/`)
> - Additional documentation (`docs/`)

## Quick Start

> **Note:** This is the **runtime-only** branch. For setup tools, data processing, and development tools, see the `setup` branch.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama (Local LLM)

Download and install Ollama from: https://ollama.ai/

Pull the recommended model:
```bash
ollama pull gemma2:2b
```

### 3. Prepare Data

**Prerequisites:**
- Processed tickets file: `data/processed/processed_tickets.json`
- Guides file: `data/guides/guides.json`
- Vector database: `data/chroma/` (populated)

> **To set up data:** Switch to the `setup` branch and run the setup scripts.

### 4. Launch the Application

```bash
python scripts/run_streamlit.py
```

Or directly:
```bash
streamlit run streamlit_app.py --server.port 8501
```

Access the app at: http://localhost:8501

## Features

### Multi-Draft Generation
- Generate 1-3 draft responses simultaneously
- Varying creativity levels (Conservative, Balanced, Creative)
- Select the best draft for your needs

### Smart Caching
- Instant responses for common queries
- 24-hour cache TTL
- Automatic cache management

### Configurable Retrieval
- Adjust number of relevant tickets (1-5)
- Adjust number of relevant guides (1-5)
- Real-time context preview

### Response Time Tracking
- View generation time for each response
- Performance metrics in sidebar
- Cache hit indicators

### Model Selection
- `gemma2:2b` - Fast (8s single draft, ~13s for 3 drafts)
- `qwen2.5:7b-instruct` - Balanced quality
- `llama3.1:8b` - Alternative option
- `qwen2.5:14b` - High quality (slower)

## Configuration

The `.env` file in the root directory contains:

```env
# Data paths
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
GUIDES_DATA_DIR=./data/guides

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## Data Management

> **Note:** This branch is runtime-only. To process new tickets or update guides, switch to the `setup` branch.

### Switching to Setup Branch

```bash
git checkout setup
```

The `setup` branch contains:
- Ticket processing scripts (`src/phase2/`)
- Guide scraping tools (`src/phase3/`)
- Database setup scripts (`scripts/run_phase4_setup.py`)
- Tests and diagnostics
- Additional documentation

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## How It Works

1. **User Query**: Operator pastes customer question
2. **Embedding**: Query is converted to vector embedding
3. **Retrieval**: ChromaDB finds relevant tickets & guides
4. **Context**: Retrieved content is formatted as context
5. **Prompt**: System prompt + context + query assembled
6. **Generation**: Local LLM generates response(s)
7. **Display**: Operator reviews and selects best draft

## Performance

- **Single Draft**: ~8 seconds (gemma2:2b)
- **3 Drafts (Parallel)**: ~13 seconds (gemma2:2b)
- **Cache Hit**: <0.01 seconds (instant)
- **Knowledge Base**: 19 tickets + 83 guide sections

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (local inference)
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Language**: Python 3.8+
- **Web Scraping**: aiohttp, cloudscraper, BeautifulSoup4

## Troubleshooting

### Streamlit won't start
```bash
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Kill the process if needed (replace PID)
taskkill /PID <PID> /F
```

### Model not found
```bash
# Check available models
ollama list

# Pull missing model
ollama pull gemma2:2b
```

### ChromaDB errors
```bash
# Delete and recreate database
rm -rf data/chroma/*
python scripts/run_phase4_setup.py
```

## Project Status

- Knowledge base: 15 guides (83 sections) + 2,590 tickets
- Response time: 8-13 seconds (3 drafts)
- Multi-draft generation: Active
- Caching: Active
- Production ready: Yes

## License

Internal use only - LaCuraDellAuto
