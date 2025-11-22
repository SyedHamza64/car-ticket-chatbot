# Deployment Guide

## Production Setup

This guide will help you deploy the LaCuraDellAuto AI Support Assistant to production.

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running (https://ollama.ai/)
3. **Git** (for version control)

### Initial Setup

1. **Clone/Download the repository**
   ```bash
   git clone <repository-url>
   cd chat-bot-ticket
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama models**
   ```bash
   # Recommended models (choose one or more):
   ollama pull mistral:7b-instruct    # Best quality
   ollama pull gemma2:2b              # Fastest
   ollama pull qwen2.5:7b-instruct    # Balanced
   ```

4. **Configure environment**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your settings (optional - defaults work for most cases)
   ```

5. **Prepare data**
   - Place your Zendesk export JSON file in `data/raw/export_combined.json`
   - Or use the existing processed data

6. **Initialize vector database**
   ```bash
   python scripts/run_phase4_setup.py
   ```
   This will:
   - Process tickets from `data/raw/export_combined.json`
   - Load guides from `data/guides/`
   - Create embeddings and populate ChromaDB

7. **Launch the application**
   ```bash
   python scripts/run_streamlit.py
   ```
   
   Or directly:
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```

8. **Access the application**
   - Open browser: http://localhost:8501

### Updating Data

**To add new tickets:**
1. Export new tickets from Zendesk
2. Merge with existing `data/raw/export_combined.json`
3. Run: `python -c "from src.phase2.process_tickets import TicketProcessor; TicketProcessor().process_all()"`
4. Re-populate database: `python scripts/run_phase4_setup.py`

**To update guides:**
1. Run: `python -c "from src.phase3.scrape_guides_fast import GuideScraper; import asyncio; asyncio.run(GuideScraper().scrape_all())"`
2. Re-populate database: `python scripts/run_phase4_setup.py`

### Production Considerations

- **Data files are excluded from git** (see `.gitignore`)
- **Environment variables** should be set via `.env` file (not committed)
- **Logs** are stored in `logs/` directory
- **Vector database** is stored in `data/chroma/` (persistent)

### Troubleshooting

- **Port 8501 in use**: Change port in `scripts/run_streamlit.py` or kill existing process
- **Model not found**: Pull the model with `ollama pull <model-name>`
- **Database errors**: Delete `data/chroma/` and re-run setup

### File Structure

```
chat-bot-ticket/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── config/                  # Configuration
├── src/                     # Source code
├── scripts/                 # Utility scripts
├── data/                    # Data files (not in git)
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

