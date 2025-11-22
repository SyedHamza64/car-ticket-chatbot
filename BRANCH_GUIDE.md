# Branch Guide

This repository has **two branches** for different use cases:

## 🌿 Branch Structure

### **`main` Branch** - Runtime Only
**Purpose:** Production-ready code with only what's needed to **run** the application.

**Contains:**
- ✅ `streamlit_app.py` - Main application
- ✅ `src/phase4/` - RAG pipeline (runtime)
- ✅ `src/utils/` - Utilities (logger, model_checker)
- ✅ `config/` - Configuration
- ✅ `scripts/run_streamlit.py` - App launcher
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation

**Does NOT contain:**
- ❌ Setup scripts
- ❌ Data processing tools
- ❌ Tests
- ❌ Diagnostics
- ❌ Development tools

**Use when:**
- Deploying to production
- Client only needs to run the app
- Data is already processed and vector DB is ready

---

### **`setup` Branch** - Complete Development
**Purpose:** Full codebase with all setup tools, tests, and diagnostics.

**Contains everything in `main` PLUS:**
- 🔧 `src/phase2/` - Ticket processing
- 🔧 `src/phase3/` - Guide scraping
- 🔧 `src/phase4/populate_vector_db.py` - Database setup
- 🔧 `scripts/run_phase4_setup.py` - Setup launcher
- 🧪 `tests/` - Unit tests
- 🔍 `diagnostics/` - Diagnostic tools
- 📚 `docs/` - Additional documentation
- 🛠️ `scripts/benchmark_models.py` - Benchmarking
- 🛠️ `scripts/test_*.ps1` - Test scripts

**Use when:**
- Setting up new data
- Processing new tickets
- Updating guides
- Running tests
- Development and debugging

---

## 🔄 Switching Between Branches

### Switch to Setup Branch (for data processing)
```bash
git checkout setup
```

### Switch to Main Branch (for runtime)
```bash
git checkout main
```

---

## 📋 Workflow

### Initial Setup (First Time)
1. Clone repository: `git clone https://github.com/SyedHamza64/car-ticket-chatbot.git`
2. Switch to setup branch: `git checkout setup`
3. Process tickets: `python -c "from src.phase2.process_tickets import TicketProcessor; TicketProcessor().process_all()"`
4. Populate DB: `python scripts/run_phase4_setup.py`
5. Switch to main: `git checkout main`
6. Run app: `python scripts/run_streamlit.py`

### Adding New Data (Later)
1. Switch to setup branch: `git checkout setup`
2. Process new tickets
3. Re-populate database
4. Switch back to main: `git checkout main`
5. Run app

### Production Deployment
1. Use `main` branch (already clean)
2. Ensure data files are in place:
   - `data/processed/processed_tickets.json`
   - `data/guides/guides.json`
   - `data/chroma/` (vector database)
3. Run: `python scripts/run_streamlit.py`

---

## 🎯 Summary

| Feature | `main` Branch | `setup` Branch |
|---------|--------------|-----------------|
| Runtime Files | ✅ | ✅ |
| Setup Scripts | ❌ | ✅ |
| Tests | ❌ | ✅ |
| Diagnostics | ❌ | ✅ |
| Documentation | Basic | Complete |
| Production Ready | ✅ | ✅ |
| Development Tools | ❌ | ✅ |

**Recommendation:**
- **Clients:** Use `main` branch for deployment
- **Developers:** Use `setup` branch for data management and development

