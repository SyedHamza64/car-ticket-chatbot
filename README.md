# LaCuraDellAuto AI Support Assistant

AI-powered support assistant for LaCuraDellAuto customer service team using RAG (Retrieval-Augmented Generation) architecture.

## Overview

This system helps support agents draft accurate, on-brand customer responses by:
- Learning from 2,590 historical Zendesk tickets
- Referencing 15 official LCDA technical guides (83 sections)
- Maintaining brand tone and consistency
- Generating multiple draft variations for operator selection

## Project Structure

```
chat-bot-ticket/
├── streamlit_app.py       # Main web interface
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── pytest.ini            # Pytest configuration
├── config/
│   └── settings.py        # Configuration settings
├── src/
│   ├── phase2/
│   │   └── process_tickets.py      # Ticket data processing
│   ├── phase3/
│   │   └── scrape_guides_fast.py   # Web scraper (optimized)
│   ├── phase4/
│   │   ├── vector_db.py            # ChromaDB vector database
│   │   ├── rag_pipeline.py         # RAG orchestration
│   │   └── populate_vector_db.py   # Database setup
│   └── utils/
│       ├── logger.py               # Logging utilities
│       └── model_checker.py        # Model validation
├── scripts/
│   ├── run_streamlit.py            # Streamlit launcher
│   ├── run_phase4_setup.py         # Database initialization
│   ├── benchmark_models.py         # Model benchmarking
│   ├── test_ollama_api.ps1         # Ollama API tests
│   └── test_models_comparison.ps1  # Model comparison tests
├── data/
│   ├── raw/                        # Raw Zendesk exports (export_combined.json)
│   ├── processed/                  # Processed tickets
│   ├── guides/                     # Scraped guides
│   └── chroma/                     # Vector database
├── tests/                          # Unit tests
├── diagnostics/                    # Diagnostic tools and results
├── docs/                           # Documentation files
└── logs/                           # Application logs
```

## Quick Start

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

### 3. Initialize Vector Database (First Time Only)

```bash
python scripts/run_phase4_setup.py
```

This will:
- Load processed tickets into ChromaDB
- Load scraped guides into ChromaDB
- Create embeddings for semantic search

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

## Utilities

### Process New Tickets

If you have new Zendesk ticket exports:

```bash
# Place the JSON file in data/raw/
python -c "from src.phase2.process_tickets import TicketProcessor; TicketProcessor().process()"

# Then re-populate the database
python scripts/run_phase4_setup.py
```

### Re-scrape Guides

If guides on the website are updated:

```bash
python -c "from src.phase3.scrape_guides_fast import GuideScraper; import asyncio; asyncio.run(GuideScraper().scrape_all())"

# Then re-populate the database
python scripts/run_phase4_setup.py
```

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
