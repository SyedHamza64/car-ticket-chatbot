# Complete Workflow Analysis

## 🔄 WORKFLOW: From Tickets to Response

### **SETUP PHASE (One-time, when adding new data)**

```
1. Raw Tickets (data/raw/export_combined.json)
   ↓
   src/phase2/process_tickets.py
   ↓
   Processed Tickets (data/processed/processed_tickets.json)

2. Guides Scraping (optional - if guides need updating)
   ↓
   src/phase3/scrape_guides_fast.py
   ↓
   Guides (data/guides/guides.json)

3. Populate Vector Database
   ↓
   scripts/run_phase4_setup.py
   → src/phase4/populate_vector_db.py
   → src/phase4/vector_db.py
   ↓
   Vector DB (data/chroma/) with embeddings
```

### **RUNTIME PHASE (Every user query)**

```
User Query
   ↓
streamlit_app.py (UI)
   ↓
src/phase4/rag_pipeline.py (Orchestrator)
   ├─→ src/phase4/vector_db.py (Retrieval from ChromaDB)
   └─→ ollama API (LLM Generation)
   ↓
Response to User
```

---

## 📁 FILE CATEGORIZATION

### ✅ **ESSENTIAL (Runtime - Required for app to work)**

**Core Application:**

- `streamlit_app.py` - Main UI application
- `config/` - Configuration (paths, settings)
- `src/phase4/` - RAG pipeline (runtime)
  - `rag_pipeline.py` - Main orchestrator
  - `vector_db.py` - Vector database operations
- `src/utils/` - Utilities
  - `logger.py` - Logging
  - `model_checker.py` - Model availability check
- `scripts/run_streamlit.py` - App launcher
- `requirements.txt` - Dependencies
- `README.md` - Documentation

**Data (already excluded from git):**

- `data/chroma/` - Vector database (runtime)
- `data/processed/processed_tickets.json` - Processed tickets (runtime)
- `data/guides/guides.json` - Guides (runtime)

---

### 🔧 **SETUP ONLY (Needed only when adding/updating data)**

**Data Processing:**

- `src/phase2/process_tickets.py` - Process raw tickets
- `scripts/run_phase4_setup.py` - Populate vector DB
- `src/phase4/populate_vector_db.py` - DB population logic

**Optional Data Updates:**

- `src/phase3/scrape_guides_fast.py` - Scrape guides (only if guides need updating)

---

### ❌ **NOT ESSENTIAL (Can remove for production)**

**Testing & Diagnostics:**

- `tests/` - Unit tests (development only)
- `diagnostics/` - Diagnostic tools (development only)
- `scripts/benchmark_models.py` - Benchmarking (development only)
- `scripts/test_*.ps1` - Test scripts (development only)
- `pytest.ini` - Test configuration

**Documentation (Optional - nice to have but not required):**

- `docs/` - Additional documentation
- `DEPLOYMENT.md` - Deployment guide
- `PRODUCTION_READY.md` - Production checklist

**Other:**

- `setup.py` - Package setup (not needed for app)
- `.streamlit/` - Streamlit config (optional)

---

## 🎯 RECOMMENDED PRODUCTION STRUCTURE

### **Minimal (Runtime Only):**

```
chat-bot-ticket/
├── streamlit_app.py          ✅ Essential
├── requirements.txt           ✅ Essential
├── README.md                  ✅ Essential
├── .env.example              ✅ Essential
├── .gitignore                ✅ Essential
├── config/                   ✅ Essential
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── phase4/               ✅ Essential (runtime)
│   │   ├── rag_pipeline.py
│   │   └── vector_db.py
│   └── utils/                ✅ Essential (runtime)
│       ├── logger.py
│       └── model_checker.py
└── scripts/
    └── run_streamlit.py      ✅ Essential
```

### **Recommended (Includes Setup Tools):**

```
chat-bot-ticket/
├── [Minimal files above]
├── src/
│   ├── phase2/               🔧 Setup only
│   │   └── process_tickets.py
│   ├── phase4/
│   │   └── populate_vector_db.py  🔧 Setup only
│   └── phase3/                🔧 Optional (if guides need updating)
│       └── scrape_guides_fast.py
└── scripts/
    └── run_phase4_setup.py   🔧 Setup only
```

---

## 🗑️ FILES TO REMOVE FOR PRODUCTION

1. **`tests/`** - Unit tests (not needed in production)
2. **`diagnostics/`** - Diagnostic tools (development only)
3. **`scripts/benchmark_models.py`** - Benchmarking tool
4. **`scripts/test_*.ps1`** - Test scripts
5. **`pytest.ini`** - Test configuration
6. **`setup.py`** - Not needed for app execution
7. **`docs/`** - Optional (can keep README only)

---

## 💡 RECOMMENDATION

**Keep:**

- ✅ All runtime files (streamlit_app.py, phase4, utils, config)
- ✅ Setup files (phase2, populate_vector_db, run_phase4_setup)
- ✅ README.md
- ✅ requirements.txt
- ✅ .env.example

**Remove:**

- ❌ tests/
- ❌ diagnostics/
- ❌ scripts/benchmark_models.py
- ❌ scripts/test\_\*.ps1
- ❌ pytest.ini
- ❌ setup.py (optional)
- ❌ docs/ (optional - or keep minimal)

This gives you a clean, production-ready codebase with only what's needed!
