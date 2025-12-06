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

# LangChain RAG pipeline
from src.phase4.rag_pipeline_langchain import LangchainRAG
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
        --response-bg: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        --response-border: #4338ca;
        --response-text: #e0e7ff;
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
        --response-bg: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        --response-border: #7dd3fc;
        --response-text: #0c4a6e;
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
        background: var(--response-bg);
        border: 1px solid var(--response-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .response-text {
        color: var(--response-text);
        font-size: 1rem;
        line-height: 1.8;
    }
    
    /* Links in response box - style Streamlit markdown links */
    .response-box a,
    .response-box .stMarkdown a,
    div[data-testid="stMarkdownContainer"] a {
        color: var(--accent-light) !important;
        text-decoration: underline !important;
        font-weight: 500 !important;
    }
    
    .response-box a:hover,
    .response-box .stMarkdown a:hover,
    div[data-testid="stMarkdownContainer"] a:hover {
        color: var(--accent) !important;
        text-decoration: underline !important;
    }
    
    /* Ensure markdown content inside response box has correct text color */
    .response-box .stMarkdown,
    .response-box .stMarkdown p,
    .response-box .stMarkdown li {
        color: var(--response-text) !important;
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
    st.session_state.current_provider = None
    st.session_state.stats = {'tickets': 0, 'guides': 0}
    st.session_state.query_history = []
    st.session_state.current_response = None
    st.session_state.current_context = None
    st.session_state.dark_mode = True  # Default to dark mode

# Initialize RAG Pipeline (cached)
@st.cache_resource
def initialize_pipeline(model_name, provider="ollama"):
    """Initialize the RAG pipeline."""
    try:
        import os
        if provider == "ollama":
            os.environ["OLLAMA_MODEL"] = model_name
        elif provider == "grok":
            os.environ["GROQ_MODEL"] = model_name
        elif provider == "gemini":
            os.environ["GEMINI_MODEL"] = model_name
        
        for module in ['config.settings', 'src.phase4.rag_pipeline_langchain', 'src.phase4']:
            if module in sys.modules:
                del sys.modules[module]
        
        from src.phase4.rag_pipeline_langchain import LangchainRAG
        pipeline = LangchainRAG(provider=provider, model=model_name)

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
    
    # Provider Selection
    provider = st.radio(
        "🔌 **AI Provider**",
        ["🦙 Ollama (Local)", "⚡ Groq (Cloud)", "✨ Gemini (Google AI)"],
        index=0,
        help="Choose between local Ollama, Groq Cloud, or Google Gemini"
    )
    if "Ollama" in provider:
        provider_name = "ollama"
    elif "Groq" in provider:
        provider_name = "grok"
    else:
        provider_name = "gemini"
    
    # Model Selection based on provider
    if provider_name == "grok":
        # Groq model names (fast inference models)
        AVAILABLE_MODELS_GROQ = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "groq/compound",
            "groq/compound-mini",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
        ]
        selected_model = st.selectbox(
            "⚡ **Groq Model**",
            AVAILABLE_MODELS_GROQ,
            index=0,
            help="Select Groq model (fast inference)"
        )
    elif provider_name == "gemini":
        # Gemini model ids (Google AI Studio)
        # Use full ids compatible with v1 generateContent (from ListModels)
        AVAILABLE_MODELS_GEMINI = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-2.0-flash-exp",
        ]
        selected_model = st.selectbox(
            "✨ **Gemini Model**",
            AVAILABLE_MODELS_GEMINI,
            index=0,
            help="Select Gemini model (Google AI Studio)"
        )
    else:
        selected_model = st.selectbox(
            "🤖 **Ollama Model**",
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
    provider_changed = st.session_state.get('current_provider') != provider_name
    model_changed = st.session_state.current_model != selected_model

    if not st.session_state.initialized or model_changed or provider_changed:
        with st.spinner(f"Loading {provider_name}/{selected_model}..."):
            if (model_changed or provider_changed) and st.session_state.initialized:
                st.cache_resource.clear()
            
            pipeline, error = initialize_pipeline(selected_model, provider=provider_name)
            if error:
                st.error(f"❌ {error}")
                st.stop()
            else:
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.session_state.current_model = selected_model
                st.session_state.current_provider = provider_name
                try:
                    st.session_state.stats = pipeline.get_stats()
                except Exception as e:
                    st.error(f"Error getting stats: {e}")
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
    provider_icons = {
        'ollama': '🦙 Ollama',
        'grok': '⚡ Groq',
        'gemini': '✨ Gemini'
    }
    provider_display = provider_icons.get(st.session_state.get('current_provider', 'ollama'), '🤖 Unknown')
    st.caption(f"Provider: {provider_display} | Model: `{st.session_state.current_model}`")

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
                result = st.session_state.pipeline.answer(
                    query,
                    top_k_tickets=n_tickets,
                    top_k_guides=n_guides,
                )
                elapsed = time.time() - start_time
                
                # Map answer() response to expected format
                st.session_state.current_response = result['answer']  # answer() returns 'answer', not 'response'
                st.session_state.current_responses = None  # answer() doesn't support multiple drafts
                st.session_state.num_drafts = 1  # Single response only
                st.session_state.current_context = result['context']  # Formatted context string
                st.session_state.current_sources = result['sources']  # Store sources separately for display
                st.session_state.response_time = elapsed
                
                st.session_state.query_history.append({
                    'time': datetime.now().strftime("%H:%M"),
                    'query': query[:50],
                    'response': result['answer']  # answer() returns 'answer', not 'response'
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
            # Single response - render markdown with proper link support
            # Use Streamlit's markdown renderer which handles links automatically
            st.markdown("""
            <div class="response-box">
            """, unsafe_allow_html=True)
            # Render markdown - this will make [text](url) links clickable
            st.markdown(st.session_state.current_response)
            st.markdown("""
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
            
            # Tickets source view
            with src_tab1:
                if st.session_state.get('current_sources'):
                    tickets = st.session_state.current_sources.get('tickets', {})
                    if tickets.get('ids') and tickets['ids'][0]:
                        docs = tickets.get('documents', [[]])[0] or []
                        metas = tickets.get('metadatas', [[]])[0] or []
                        dists = tickets.get('distances', [[]])[0] or []
                        
                        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                            subject = meta.get('subject', 'N/A')
                            status = meta.get('status', 'N/A')
                            ticket_id = meta.get('ticket_id') or meta.get('orig_ticket_id', 'N/A')  # Check both fields
                            priority = meta.get('priority', 'N/A')
                            created_at = meta.get('created_at', 'N/A')
                            
                            # Relevance badge based on distance (lower distance = higher relevance)
                            relevance_badge = ""
                            try:
                                dist = float(dists[i-1]) if len(dists) >= i else None
                            except Exception:
                                dist = None
                            
                            if dist is not None:
                                if dist <= 0.25:
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.5:
                                    relevance_badge = "🟡 **Medium relevance**"
                                else:
                                    relevance_badge = "🔴 **Low relevance**"
                            
                            st.markdown(f"**{i}. {subject}**")
                            caption_line = f"Status: {status} • Ticket ID: {ticket_id}"
                            if relevance_badge:
                                caption_line += f" • {relevance_badge}"
                            st.caption(caption_line)
                            
                            if st.checkbox("Show details", key=f"ticket_src_{i}"):
                                # Structured metadata
                                st.markdown(f"- **Ticket ID**: `{ticket_id}`")
                                st.markdown(f"- **Status**: `{status}`")
                                if priority and priority != 'N/A':
                                    st.markdown(f"- **Priority**: `{priority}`")
                                if created_at and created_at != 'N/A':
                                    st.markdown(f"- **Created at**: `{created_at}`")
                                
                                # Parse searchable_text into subject, description, conversation
                                from html import unescape
                                subject_text = ""
                                description_text = ""
                                messages = []
                                for line in (doc or "").splitlines():
                                    raw = line.strip()
                                    if not raw:
                                        continue
                                    if raw.startswith("Subject:"):
                                        subject_text = raw[len("Subject:"):].strip()
                                    elif raw.startswith("Description:"):
                                        description_text = raw[len("Description:"):].strip()
                                    elif ": " in raw:
                                        author, msg = raw.split(": ", 1)
                                        messages.append((author.strip(), unescape(msg.strip())))
                                
                                st.markdown("---")
                                if subject_text:
                                    st.markdown(f"**Subject**: {subject_text}")
                                if description_text:
                                    st.markdown(f"**Description**: {description_text}")
                                
                                if messages:
                                    st.markdown("**Conversation:**")
                                    for author, msg in messages:
                                        # Show a reasonable preview per message
                                        preview = msg if len(msg) <= 500 else msg[:500] + " [...]"
                                        st.markdown(f"- **{author}**: {preview}")
                                
                                # Optional raw view (use checkbox instead of nested expander)
                                if st.checkbox("Show raw searchable text", key=f"ticket_raw_{i}"):
                                    st.text(doc if len(doc) <= 2000 else doc[:2000] + "\n...\n[truncated]")
                            
                            st.markdown("---")
                    else:
                        st.info("No tickets found")
                else:
                    st.info("No tickets found")
            
            # Guides source view
            with src_tab2:
                if st.session_state.get('current_sources'):
                    guides = st.session_state.current_sources.get('guides', {})
                    if guides.get('ids') and guides['ids'][0]:
                        docs = guides.get('documents', [[]])[0] or []
                        metas = guides.get('metadatas', [[]])[0] or []
                        dists = guides.get('distances', [[]])[0] or []
                        
                        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                            guide_title = meta.get('guide_title', 'N/A')
                            section_title = meta.get('section_title', 'N/A')
                            guide_url = meta.get('url', '')
                            guide_number = meta.get('guide_number', '')
                            
                            # Relevance badge based on distance
                            relevance_badge = ""
                            try:
                                dist = float(dists[i-1]) if len(dists) >= i else None
                            except Exception:
                                dist = None
                            
                            if dist is not None:
                                if dist <= 0.25:
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.5:
                                    relevance_badge = "🟡 **Medium relevance**"
                                else:
                                    relevance_badge = "🔴 **Low relevance**"
                            
                            # Display guide title with link if available
                            if guide_url and guide_url != 'N/A' and guide_url.strip():
                                st.markdown(f"**{i}. [{guide_title}]({guide_url})** 🔗")
                            else:
                                st.markdown(f"**{i}. {guide_title}**")
                            
                            caption_parts = []
                            if guide_number and guide_number != 'N/A':
                                caption_parts.append(f"Guide: {guide_number}")
                            if section_title and section_title != 'N/A':
                                caption_parts.append(f"Section: {section_title}")
                            if relevance_badge:
                                caption_parts.append(relevance_badge)
                            if caption_parts:
                                st.caption(" • ".join(caption_parts))
                            
                            if st.checkbox("Show content", key=f"guide_src_{i}"):
                                st.text(doc if len(doc) <= 2000 else doc[:2000] + "\n...\n[truncated]")
                            
                            st.markdown("---")
                    else:
                        st.info("No guides found")
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
# HELPER FUNCTION: Run incremental update scripts
# ============================================================================
def run_fast_update(script):
    """Run an incremental update script and stream output to Streamlit."""
    with st.status(f"Running `{script}`...", expanded=True) as status:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root
        )

        for line in process.stdout:
            status.write(line.strip())

        process.wait()

        if process.returncode == 0:
            status.update(label="Completed!", state="complete")
            # Refresh stats after update
            try:
                st.session_state.stats = st.session_state.pipeline.get_stats()
            except:
                pass
        else:
            status.update(label="Failed", state="error")

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
        
        # Quick Update Section (Incremental)
        st.markdown("#### ⚡ **Quick Updates (Incremental)**")
        st.markdown("*Fast incremental updates - only processes new/changed items*")
        
        # Update Tickets Only (Fast)
        if st.button("📋 Update Tickets Only (Fast)", use_container_width=True):
            run_fast_update("scripts/update_tickets_only.py")
        
        # Update Guides Only (Fast)
        if st.button("📘 Update Guides Only (Fast)", use_container_width=True):
            run_fast_update("scripts/update_guides_incremental.py")
        
        # Update All (Fast)
        if st.button("🔄 Update All (Fast)", use_container_width=True):
            run_fast_update("scripts/update_tickets_only.py")
            run_fast_update("scripts/update_guides_incremental.py")
        
        st.markdown("---")
        
        # Full Rebuild (for emergencies)
        st.markdown("#### 🔨 **Full Rebuild**")
        st.markdown("*Complete database recreation (2-5 min)*")
        
        if st.button("🔨 Full Rebuild", use_container_width=True):
            with st.status("Rebuilding database...", expanded=True) as status:
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/rebuild_vector_db_v2.py"],
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
provider_name = st.session_state.get('current_provider', 'ollama')
provider_label = "Ollama" if provider_name == "ollama" else "Groq"
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">
    Built with ❤️ for LaCuraDellAuto • Powered by {provider_label} ({st.session_state.current_model or 'AI'})
</div>
""", unsafe_allow_html=True)
