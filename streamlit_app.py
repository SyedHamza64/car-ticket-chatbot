"""
LaCuraDellAuto AI Support Assistant
Modern, Clean, Professional Interface
"""

import streamlit as st
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase4.rag_pipeline import RAGPipeline
from src.phase4.vector_db import VectorDBManager
from src.utils.model_checker import get_available_models

# Page configuration
st.set_page_config(
    page_title="LaCuraDellAuto AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Get current theme
is_dark = st.session_state.dark_mode

# Dynamic CSS variables based on theme
if is_dark:
    css_vars = """
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --bg-hover: #22222e;
        --accent: #6366f1;
        --accent-light: #a5b4fc;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #ffffff;
        --text-secondary: #e2e8f0;
        --text-muted: #94a3b8;
        --text-heading: #c7d2fe;
        --border: #2a2a3a;
        --success: #34d399;
        --warning: #fbbf24;
        --error: #f87171;
        --input-bg: #12121a;
        --input-text: #ffffff;
    }
    """
else:
    css_vars = """
    :root {
        --bg-primary: #f8fafc;
        --bg-secondary: #f1f5f9;
        --bg-card: #ffffff;
        --bg-hover: #e2e8f0;
        --accent: #6366f1;
        --accent-light: #4f46e5;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --text-heading: #4f46e5;
        --border: #e2e8f0;
        --success: #34d399;
        --warning: #fbbf24;
        --error: #f87171;
        --input-bg: #ffffff;
        --input-text: #1e293b;
    }
    """

# Modern CSS - Dynamic theme with accent colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    """ + css_vars + """
    
    * {
        font-family: 'Space Grotesk', -apple-system, sans-serif;
    }
    
    code, .stCode, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main app background */
    .stApp {
        background: var(--bg-primary);
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light), #a78bfa);
    }
    
    .app-header h1 {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
    }
    
    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }
    
    .card-title {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Stats */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-box {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        color: var(--accent-light);
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    /* Response area */
    .response-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .response-text {
        color: #e0e7ff;
        font-size: 1rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }
    
    /* ALL Buttons - Force dark theme */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"],
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"],
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.25rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px var(--accent-glow) !important;
    }
    
    .stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        color: white !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    .stButton > button:focus,
    button:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    
    /* Button text color fix */
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: white !important;
    }
    
    /* Text inputs */
    .stTextArea textarea, .stTextInput input, .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }
    
    /* Labels */
    .stTextArea label, .stTextInput label, .stSelectbox label, .stRadio label {
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-heading) !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }
    
    /* Main headings */
    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown h3, .stMarkdown h4 {
        color: var(--text-heading) !important;
        font-weight: 600 !important;
    }
    
    /* Radio buttons text */
    .stRadio > div > label {
        color: var(--text-primary) !important;
    }
    
    /* Caption text - make it brighter */
    .stCaption, small, .stMarkdown small {
        color: var(--text-muted) !important;
    }
    
    /* Italic text */
    em, i, .stMarkdown em {
        color: var(--text-secondary) !important;
        font-style: italic;
    }
    
    /* Bold text */
    strong, b, .stMarkdown strong {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Paragraph text */
    p, .stMarkdown p {
        color: var(--text-secondary) !important;
    }
    
    /* Code inline */
    code {
        background: var(--bg-hover) !important;
        color: var(--accent-light) !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent-light) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* Form labels */
    .stForm label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Form submit button */
    .stForm [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%) !important;
        color: white !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }
    
    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    
    /* Tooltips / Help text */
    .stTooltipIcon {
        color: var(--text-muted) !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border);
        margin: 1.5rem 0;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) transparent transparent transparent;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-info {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 9999;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .theme-toggle:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 20px var(--accent-glow);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pipeline = None
    st.session_state.current_model = None
    st.session_state.stats = {'tickets': 0, 'guides': 0}
    st.session_state.query_history = []
    st.session_state.current_response = None
    st.session_state.current_context = None
    st.session_state.dark_mode = True  # Default to dark mode

# Initialize RAG Pipeline (cached)
@st.cache_resource
def initialize_pipeline(model_name):
    """Initialize the RAG pipeline."""
    try:
        import os
        os.environ['OLLAMA_MODEL'] = model_name
        
        for module in ['config.settings', 'src.phase4.rag_pipeline', 'src.phase4']:
            if module in sys.modules:
                del sys.modules[module]
        
        from src.phase4.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(model=model_name)
        return pipeline, None
    except Exception as e:
        return None, str(e)

# Get available models
try:
    installed_models = get_available_models()
    AVAILABLE_MODELS = installed_models if installed_models else ['gemma2:2b', 'llama3.1:8b']
except:
    AVAILABLE_MODELS = ['gemma2:2b', 'llama3.1:8b']

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ **Settings**")
    
    # Theme Toggle
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.markdown("**🎨 Theme**")
    with theme_col2:
        if st.button("🌙" if st.session_state.dark_mode else "☀️", key="theme_toggle", help="Toggle Light/Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("---")
    
    # Model Selection
    selected_model = st.selectbox(
        "🤖 **AI Model**",
        AVAILABLE_MODELS,
        index=0,
        help="Select Ollama model"
    )
    
    # Language
    st.markdown("---")
    language = st.radio(
        "🌐 **Response Language**",
        ["🇮🇹 Italiano", "🇬🇧 English"],
        index=0
    )
    language_code = "italian" if "Italiano" in language else "english"
    
    # Initialize model
    st.markdown("---")
    model_changed = st.session_state.current_model != selected_model

    if not st.session_state.initialized or model_changed:
        with st.spinner(f"Loading {selected_model}..."):
            if model_changed and st.session_state.initialized:
                st.cache_resource.clear()
            
            pipeline, error = initialize_pipeline(selected_model)
            if error:
                st.error(f"❌ {error}")
                st.stop()
            else:
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.session_state.current_model = selected_model
                try:
                    st.session_state.stats = pipeline.db_manager.get_stats()
                except:
                    st.session_state.stats = {'tickets': 0, 'guides': 0}

    # Stats
    st.markdown("### 📊 **Knowledge Base**")
    col1, col2 = st.columns(2)
    col1.metric("Tickets", st.session_state.stats['tickets'])
    col2.metric("Guides", st.session_state.stats['guides'])
    
    # Actions
    st.markdown("---")
    st.markdown("### ⚡ **Actions**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.query_history = []
            st.session_state.current_response = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption(f"Model: `{st.session_state.current_model}`")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
    st.markdown("""
<div class="app-header">
    <h1>🚗 LaCuraDellAuto AI</h1>
    <p>Intelligent Customer Support Assistant</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab_query, tab_manage = st.tabs(["💬 Ask Question", "🗄️ Manage Knowledge Base"])

# ============================================================================
# TAB 1: QUERY INTERFACE
# ============================================================================
with tab_query:
    # Query input
    st.markdown("### 💬 **What would you like to know?**")
    
    query = st.text_area(
        "Enter your question",
        height=100,
        placeholder="Es: Come posso rimuovere i graffi dalla carrozzeria?\nEs: Quale prodotto usare per lucidare l'auto?",
        label_visibility="collapsed"
    )

    # Options row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        n_tickets = st.selectbox("📋 Tickets", [1, 2, 3, 4, 5], index=2, label_visibility="collapsed")
    with col2:
        n_guides = st.selectbox("📚 Guides", [1, 2, 3, 4, 5], index=2, label_visibility="collapsed")
    with col3:
        num_drafts = st.selectbox("📝 Drafts", [1, 2, 3], index=0, label_visibility="collapsed")
    with col4:
        generate_btn = st.button("✨ Generate Response", type="primary", use_container_width=True)
            
    # Generate response
    if generate_btn and query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                result = st.session_state.pipeline.query(
                    query,
                    n_tickets=n_tickets,
                    n_guides=n_guides,
                    num_drafts=num_drafts,
                    language=language_code
                )
                elapsed = time.time() - start_time
                
                st.session_state.current_response = result['response']
                st.session_state.current_responses = result.get('responses', None)
                st.session_state.num_drafts = result.get('num_drafts', 1)
                st.session_state.current_context = result['context']
                st.session_state.response_time = elapsed
                
                st.session_state.query_history.append({
                    'time': datetime.now().strftime("%H:%M"),
                    'query': query[:50],
                    'response': result['response']
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display response
    if st.session_state.current_response:
        st.markdown("---")
        
        # Response header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 💡 AI Response")
        with col2:
            st.caption(f"⏱️ {st.session_state.get('response_time', 0):.1f}s")
        
        # Multiple drafts
        if st.session_state.get('num_drafts', 1) > 1 and st.session_state.get('current_responses'):
            draft_tabs = st.tabs([f"Draft {i+1}" for i in range(st.session_state.num_drafts)])
            for i, tab in enumerate(draft_tabs):
                with tab:
                    st.markdown(f"""
                    <div class="response-box">
                        <div class="response-text">{st.session_state.current_responses[i]['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Single response
            st.markdown(f"""
            <div class="response-box">
                <div class="response-text">{st.session_state.current_response}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.toast("✅ Copied!", icon="📋")
        with col2:
            if st.button("👍 Helpful", use_container_width=True):
                st.toast("Thanks for feedback!", icon="👍")
        with col3:
            if st.button("👎 Not helpful", use_container_width=True):
                st.toast("We'll improve!", icon="📝")
        
        # Sources
        with st.expander("📚 View Sources", expanded=False):
            src_tab1, src_tab2 = st.tabs(["Tickets", "Guides"])
            
            with src_tab1:
                if st.session_state.current_context:
                    tickets = st.session_state.current_context.get('tickets', {})
                    if tickets.get('ids') and tickets['ids'][0]:
                        for i, (doc, meta) in enumerate(zip(tickets['documents'][0], tickets['metadatas'][0]), 1):
                            st.markdown(f"**{i}. {meta.get('subject', 'N/A')}**")
                            st.caption(f"Status: {meta.get('status', 'N/A')}")
                            if st.checkbox(f"Show content", key=f"ticket_src_{i}"):
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                            st.markdown("---")
                    else:
                        st.info("No tickets found")
            
            with src_tab2:
                if st.session_state.current_context:
                    guides = st.session_state.current_context.get('guides', {})
                    if guides.get('ids') and guides['ids'][0]:
                        for i, (doc, meta) in enumerate(zip(guides['documents'][0], guides['metadatas'][0]), 1):
                            st.markdown(f"**{i}. {meta.get('guide_title', 'N/A')}**")
                            st.caption(f"Section: {meta.get('section_title', 'N/A')}")
                            if st.checkbox(f"Show content", key=f"guide_src_{i}"):
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                            st.markdown("---")
                    else:
                        st.info("No guides found")

    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📜 **Recent Queries**")
        
        for item in reversed(st.session_state.query_history[-3:]):
            with st.expander(f"🕐 {item['time']} — {item['query']}..."):
                st.text(item['response'][:300] + "..." if len(item['response']) > 300 else item['response'])

# ============================================================================
# TAB 2: KNOWLEDGE BASE MANAGEMENT
# ============================================================================
with tab_manage:
    st.markdown("### 🗄️ **Knowledge Base Management**")
    st.markdown("Upload tickets, refresh guides, and rebuild the vector database.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    
    # Left column
    with col1:
        # Upload Zendesk Export
        st.markdown("#### 📤 **Upload Zendesk Export**")
        st.markdown("*Upload raw Zendesk export file (NDJSON format)*")
        
        uploaded = st.file_uploader("Upload JSON", type=['json'], label_visibility="collapsed")
        if uploaded:
            try:
                # Read the uploaded file content
                content = uploaded.read().decode('utf-8')
                
                # Parse NDJSON (each line is a JSON object)
                new_tickets = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        ticket = json.loads(line)
                        new_tickets.append(ticket)
                
                st.info(f"📊 Found **{len(new_tickets)}** tickets in export")
                
                # Show preview of first ticket
                with st.expander("📄 Preview First Ticket", expanded=False):
                    if new_tickets:
                        preview = {
                            'id': new_tickets[0].get('id'),
                            'subject': new_tickets[0].get('subject'),
                            'status': new_tickets[0].get('status'),
                            'created_at': new_tickets[0].get('created_at')
                        }
                        st.json(preview)
                
                if st.button("➕ Add to Raw Data", use_container_width=True, type="primary"):
                    # Load existing combined export
                    combined_path = Path("data/raw/export_combined.json")
                    existing_tickets = []
                    existing_ids = set()
                    
                    if combined_path.exists():
                        with open(combined_path, 'r', encoding='utf-8') as f:
                            existing_tickets = json.load(f)
                            existing_ids = {t.get('id') for t in existing_tickets}
                    
                    # Add only new tickets (avoid duplicates)
                    added_count = 0
                    for ticket in new_tickets:
                        if ticket.get('id') not in existing_ids:
                            existing_tickets.append(ticket)
                            added_count += 1
                    
                    # Save updated combined file
                    with open(combined_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_tickets, f, ensure_ascii=False, indent=2)
                    
                    if added_count > 0:
                        st.success(f"✅ Added **{added_count}** new tickets to raw data!")
                        st.info("💡 Now click **Process Tickets** → **Rebuild Vector DB** to update the knowledge base")
                    else:
                        st.warning("⚠️ All tickets already exist in the database")
                        
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Refresh Guides
        st.markdown("#### 🌐 **Refresh Guides**")
        st.markdown("*Scrape latest guides from LaCuraDellAuto website*")
        
        if st.button("🔄 Scrape Guides", use_container_width=True):
            with st.status("Scraping guides...", expanded=True) as status:
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "src.phase3.scrape_guides_fast"],
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    for line in process.stdout:
                        if line.strip():
                            status.write(line.strip())
                    
                    process.wait(timeout=120)
                    
                    if process.returncode == 0:
                        status.update(label="✅ Guides scraped!", state="complete")
                    else:
                        status.update(label="❌ Scraping failed", state="error")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Right column
    with col2:
        # Process Tickets
        st.markdown("#### 🔧 **Process Tickets**")
        st.markdown("*Process raw ticket exports from* `data/raw/`")
        
        if st.button("⚙️ Process Tickets", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "src.phase2.process_tickets"],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=180
                    )
                    if result.returncode == 0:
                        st.success("✅ Tickets processed!")
                    else:
                        st.error(f"Error: {result.stderr}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Quick Update Section
        st.markdown("#### ⚡ **Quick Updates**")
        st.markdown("*Fast re-indexing without full rebuild*")
        
        # Update Tickets Only
        if st.button("📋 Update Tickets Only", use_container_width=True, help="Re-index tickets (~30 sec)"):
            with st.spinner("Updating tickets..."):
                try:
                    db = st.session_state.pipeline.db_manager
                    db.reset_tickets_collection()
                    db.add_tickets()
                    st.session_state.stats = db.get_stats()
                    st.session_state.pipeline._cache.clear()
                    st.success(f"✅ Tickets updated! ({st.session_state.stats['tickets']} indexed)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Update Guides Only
        if st.button("📚 Update Guides Only", use_container_width=True, help="Re-index guides (~10 sec)"):
            with st.spinner("Updating guides..."):
                try:
                    db = st.session_state.pipeline.db_manager
                    db.reset_guides_collection()
                    db.add_guides()
                    st.session_state.stats = db.get_stats()
                    st.session_state.pipeline._cache.clear()
                    st.success(f"✅ Guides updated! ({st.session_state.stats['guides']} chunks indexed)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Update Both (Tickets + Guides)
        if st.button("🔄 Update All (Tickets + Guides)", use_container_width=True, help="Re-index both (~1 min)"):
            with st.spinner("Updating tickets and guides..."):
                try:
                    db = st.session_state.pipeline.db_manager
                    
                    # Update tickets
                    db.reset_tickets_collection()
                    db.add_tickets()
                    
                    # Update guides
                    db.reset_guides_collection()
                    db.add_guides()
                    
                    st.session_state.stats = db.get_stats()
                    st.session_state.pipeline._cache.clear()
                    st.success(f"✅ All updated! ({st.session_state.stats['tickets']} tickets, {st.session_state.stats['guides']} guide chunks)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Full Rebuild (for emergencies)
        st.markdown("#### 🔨 **Full Rebuild**")
        st.markdown("*Complete database recreation (2-5 min)*")
        
        if st.button("🔨 Full Rebuild", use_container_width=True):
            with st.status("Rebuilding database...", expanded=True) as status:
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/rebuild_vector_db.py"],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode == 0:
                        status.update(label="✅ Database rebuilt!", state="complete")
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        status.update(label="❌ Rebuild failed", state="error")
                        st.error(result.stderr)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Database Info
        st.markdown("#### 📊 **Database Info**")
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{st.session_state.stats['tickets']}</div>
                <div class="stat-label">Tickets</div>
            </div>
            """, unsafe_allow_html=True)
        with info_col2:
                        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{st.session_state.stats['guides']}</div>
                <div class="stat-label">Guide Chunks</div>
                        </div>
                        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">
    Built with ❤️ for LaCuraDellAuto • Powered by {st.session_state.current_model or 'AI'}
</div>
""", unsafe_allow_html=True)
