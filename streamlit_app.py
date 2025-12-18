"""
LaCuraDellAuto AI Support Assistant
Modern, Clean, Professional Interface
"""

import os
import sys

# Disable CUDA by default and set other environment variables BEFORE any other imports to prevent hangs
if not os.getenv("USE_CUDA", "").lower() == "true":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Streamlit-specific environment variables
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import time
import json
from datetime import datetime
from pathlib import Path
import subprocess
import markdown

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

# Theme toggle in session state - Initialize ONCE with consistent default
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode (consistent)

# Get current theme from session state
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
    
    /* Main app background - Strong override to prevent flash */
    .stApp, .stApp > div, html, body {
        background: var(--bg-primary) !important;
        background-color: var(--bg-primary) !important;
    }
    
    /* Force main content area background */
    .main, section.main > div {
        background: var(--bg-primary) !important;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Sidebar - Force background */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem;
    }
    
    /* Sidebar toggle button - Make it VISIBLE */
    /* Target the Streamlit sidebar collapse button with all known selectors */
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"] > div:first-child > button,
    .st-emotion-cache-6qob1r button,
    button.st-emotion-cache-17zm0w6,
    button.etdmgzm15,
    .stApp > button:first-of-type {
        background: #6366f1 !important;
        color: white !important;
        border-radius: 0 8px 8px 0 !important;
        min-width: 28px !important;
        min-height: 44px !important;
        border: none !important;
        box-shadow: 3px 3px 12px rgba(0,0,0,0.5) !important;
        opacity: 1 !important;
        visibility: visible !important;
        position: fixed !important;
        left: 0 !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 999999 !important;
    }
    
    button[data-testid="stBaseButton-headerNoPadding"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    [data-testid="collapsedControl"]:hover,
    button.st-emotion-cache-17zm0w6:hover,
    button.etdmgzm15:hover {
        background: #a5b4fc !important;
        transform: translateY(-50%) scale(1.1) !important;
    }
    
    /* When sidebar is collapsed, show the expand button clearly */
    [data-testid="stSidebarCollapsedControl"] {
        background: #6366f1 !important;
        border-radius: 0 8px 8px 0 !important;
        opacity: 1 !important;
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
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98) !important;
    }
    
    /* Response box with blue border */
    .response-box {
        background: var(--response-bg);
        border: 2px solid var(--response-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .response-box p {
        color: var(--response-text) !important;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .response-box a {
        color: var(--accent) !important;
        text-decoration: underline;
        font-weight: 500;
    }
    
    .response-box a:hover {
        color: var(--accent-light) !important;
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
        caret-color: var(--text-primary) !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
        caret-color: var(--text-primary) !important;
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
    
    /* File uploader text - ensure visibility in both modes */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInput"] + div span,
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] small {
        color: var(--text-primary) !important;
    }
    
    /* Uploaded file info - file name and size */
    [data-testid="stFileUploader"] li span,
    [data-testid="stFileUploader"] li div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploader"] .uploadedFile span,
    [data-testid="stFileUploader"] .uploadedFileName,
    [data-testid="stFileUploader"] .st-emotion-cache-1aehpvj,
    [data-testid="stFileUploader"] .st-emotion-cache-nahz7x,
    section[data-testid="stFileUploader"] div:not([data-testid]) span {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
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
    # NOTE: dark_mode is initialized earlier (before CSS) to prevent flickering

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
    
    # Provider Selection (Groq only in UI, others available in backend)
    provider = st.radio(
        "🔌 **AI Provider**",
        ["⚡ Groq (Cloud)"],
        index=0,
        help="Fast, cloud-based AI inference"
    )
    # Always use Groq in UI
    provider_name = "grok"
    
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
    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    with col1:
        n_tickets = st.selectbox("📋 Ticket Sources", [1, 2, 3, 4, 5], index=4, help="Number of relevant tickets to retrieve")
    with col2:
        n_guides = st.selectbox("📚 Guide Sources", [1, 2, 3, 4, 5], index=2, help="Number of relevant guide sections to retrieve")
    with col3:
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
            # Single response - convert markdown to HTML and wrap in response-box
            response_html = markdown.markdown(st.session_state.current_response)
            st.markdown(f"""
            <div class="response-box">
                {response_html}
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
                            # Adjusted thresholds for hybrid scoring (produces higher distances than pure dense)
                            relevance_badge = ""
                            try:
                                dist = float(dists[i-1]) if len(dists) >= i else None
                            except Exception:
                                dist = None
                            
                            if dist is not None:
                                if dist <= 0.5:  # Was 0.25 - more lenient for hybrid scoring
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.75:  # Was 0.5 - medium range expanded
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
                            
                            # Relevance badge based on distance (lower distance = higher relevance)
                            # Adjusted thresholds for hybrid scoring (produces higher distances than pure dense)
                            relevance_badge = ""
                            try:
                                dist = float(dists[i-1]) if len(dists) >= i else None
                            except Exception:
                                dist = None
                            
                            if dist is not None:
                                if dist <= 0.5:  # Was 0.25 - more lenient for hybrid scoring
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.75:  # Was 0.5 - medium range expanded
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
    
    # Main upload section - simplified flow
    st.markdown("#### 📤 **Upload Zendesk Export**")
    st.markdown("*Upload raw Zendesk export file (JSON or NDJSON format)*")
    
    # Show last upload timestamp
    from config.settings import PROCESSED_TICKETS_FILE
    if PROCESSED_TICKETS_FILE.exists():
        import os
        last_modified = datetime.fromtimestamp(os.path.getmtime(PROCESSED_TICKETS_FILE))
        st.caption(f"📅 Last uploaded: {last_modified.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.caption("📅 No tickets uploaded yet")
    
    uploaded = st.file_uploader("Upload JSON", type=['json'], label_visibility="collapsed")
    
    if uploaded:
        try:
            # Read and parse file
            content = uploaded.read().decode('utf-8')
            uploaded_tickets = []
            
            try:
                # Try NDJSON format first (one JSON per line)
                for line in content.strip().split('\n'):
                    if line.strip():
                        ticket = json.loads(line)
                        uploaded_tickets.append(ticket)
            except json.JSONDecodeError:
                # If NDJSON fails, try regular JSON array
                uploaded_tickets = json.loads(content)
                if not isinstance(uploaded_tickets, list):
                    uploaded_tickets = [uploaded_tickets]
            
            st.success(f"📊 Found **{len(uploaded_tickets)}** tickets in uploaded file")
            
            # Option to clear existing tickets
            clear_existing = st.checkbox("🗑️ Clear existing tickets from knowledge base before checking", value=False, 
                                        help="If checked, all existing tickets will be removed from ChromaDB before checking for duplicates. All uploaded tickets will be treated as new.")
            
            # Check against existing KB to find unique tickets
            st.info("🔍 Checking against existing knowledge base...")
            
            try:
                from src.phase4.vector_db import VectorDBManager
                db = VectorDBManager()
                
                # Clear existing tickets if requested
                if clear_existing:
                    with st.spinner("🗑️ Clearing existing tickets from knowledge base..."):
                        if db.rag_v2_coll:
                            try:
                                # Get all ticket IDs
                                ticket_data = db.rag_v2_coll.get(where={"type": "ticket"}, limit=None)
                                ticket_ids = ticket_data.get("ids", [])
                                if ticket_ids:
                                    db.rag_v2_coll.delete(ids=ticket_ids)
                                    st.success(f"✅ Cleared {len(ticket_ids)} existing tickets from knowledge base")
                            except Exception as e:
                                st.warning(f"⚠️ Could not clear tickets: {e}")
                        elif db.tickets_coll:
                            try:
                                all_data = db.tickets_coll.get(limit=None)
                                ticket_ids = all_data.get("ids", [])
                                if ticket_ids:
                                    db.tickets_coll.delete(ids=ticket_ids)
                                    st.success(f"✅ Cleared {len(ticket_ids)} existing tickets from knowledge base")
                            except Exception as e:
                                st.warning(f"⚠️ Could not clear tickets: {e}")
                
                # Get existing ticket IDs from KB
                existing_kb_ids = set()
                if db.rag_v2_coll:
                    try:
                        existing_data = db.rag_v2_coll.get(where={"type": "ticket"}, limit=None)
                        # Extract ticket IDs from format: ticket_{id}__idx{chunk_index} or ticket_{id}
                        for id_ in existing_data.get("ids", []):
                            if id_.startswith("ticket_"):
                                # Handle both formats: ticket_123 or ticket_123__idx0
                                parts = id_.split("__idx")
                                ticket_part = parts[0]  # ticket_123
                                if "_" in ticket_part:
                                    try:
                                        ticket_id = int(ticket_part.split("_")[-1])
                                        existing_kb_ids.add(ticket_id)
                                    except ValueError:
                                        continue
                    except:
                        # Fallback: get all and filter
                        all_data = db.rag_v2_coll.get(limit=None)
                        ids = all_data.get("ids", [])
                        metas = all_data.get("metadatas", [])
                        for i, meta in enumerate(metas):
                            if meta and meta.get("type") == "ticket" and i < len(ids):
                                id_ = ids[i]
                                if id_.startswith("ticket_"):
                                    parts = id_.split("__idx")
                                    ticket_part = parts[0]
                                    if "_" in ticket_part:
                                        try:
                                            ticket_id = int(ticket_part.split("_")[-1])
                                            existing_kb_ids.add(ticket_id)
                                        except ValueError:
                                            continue
                elif db.tickets_coll:
                    # Extract ticket IDs from format: ticket_{id}__idx{chunk_index} or ticket_{id}
                    for id_ in db.tickets_coll.get()["ids"]:
                        if id_.startswith("ticket_"):
                            parts = id_.split("__idx")
                            ticket_part = parts[0]
                            if "_" in ticket_part:
                                try:
                                    ticket_id = int(ticket_part.split("_")[-1])
                                    existing_kb_ids.add(ticket_id)
                                except ValueError:
                                    continue
                
                # Find unique tickets
                unique_tickets = []
                duplicate_count = 0
                for ticket in uploaded_tickets:
                    ticket_id = ticket.get('id')
                    if ticket_id and ticket_id not in existing_kb_ids:
                        unique_tickets.append(ticket)
                    else:
                        duplicate_count += 1
                
                # Display results
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("✅ Unique Tickets", len(unique_tickets), delta=f"{len(unique_tickets)} new")
                with col_stat2:
                    st.metric("⚠️ Duplicates", duplicate_count, delta=f"{duplicate_count} already in KB")
                
                if len(unique_tickets) == 0:
                    st.warning("⚠️ All tickets already exist in the knowledge base. No new tickets to import.")
                else:
                    # Show preview
                    with st.expander("📄 Preview First Unique Ticket", expanded=False):
                        if unique_tickets:
                            preview = {
                                'id': unique_tickets[0].get('id'),
                                'subject': unique_tickets[0].get('subject'),
                                'status': unique_tickets[0].get('status'),
                                'created_at': unique_tickets[0].get('created_at')
                            }
                            st.json(preview)
                    
                    # Store unique tickets in session state for processing
                    st.session_state.unique_tickets_to_process = unique_tickets
                    st.session_state.unique_tickets_count = len(unique_tickets)
                    
                    st.success(f"💾 Ready to process **{len(unique_tickets)}** unique tickets")
                    
                    # Process button
                    st.markdown("---")
                    if st.button("🚀 Process & Update Knowledge Base", type="primary", use_container_width=True):
                        unique_tickets = st.session_state.get("unique_tickets_to_process", [])
                        if not unique_tickets:
                            st.error("No unique tickets to process. Please upload a file first.")
                            st.stop()
                        
                        with st.status("Processing and updating knowledge base...", expanded=True) as status:
                            # Step 1: Process tickets (clean, extract conversation)
                            status.write(f"📝 Step 1/2: Processing {len(unique_tickets)} unique tickets...")
                            try:
                                import tempfile
                                import os
                                
                                # Create temporary file with only unique tickets
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                                    json.dump(unique_tickets, tmp_file, ensure_ascii=False, indent=2)
                                    tmp_path = Path(tmp_file.name)
                                
                                # Set environment variable for the script to use
                                env = os.environ.copy()
                                env['ZENDESK_EXPORT_FILE'] = str(tmp_path)
                                
                                # Use Popen for real-time streaming
                                process = subprocess.Popen(
                                    [sys.executable, "scripts/process_tickets_only.py"],
                                    cwd=project_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,  # Line buffered
                                    universal_newlines=True,
                                    env=env
                                )
                                
                                # Stream output in real-time
                                for line in iter(process.stdout.readline, ''):
                                    if line:
                                        line = line.strip()
                                        if line:
                                            status.write(line)
                                
                                process.wait()
                                
                                # Clean up temp file
                                if tmp_path.exists():
                                    tmp_path.unlink()
                                
                                if process.returncode != 0:
                                    status.update(label="❌ Processing failed", state="error")
                                    st.error("Processing failed. Check logs above.")
                                    st.stop()
                                
                            except Exception as e:
                                status.write(f"❌ Error processing: {e}")
                                st.error(f"Processing failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.stop()
                            
                            # Step 2: Generate embeddings and update KB (only new tickets will be added)
                            status.write("")
                            status.write("🔍 Step 2/2: Generating embeddings and updating knowledge base...")
                            try:
                                # Use Popen for real-time streaming
                                process = subprocess.Popen(
                                    [sys.executable, "scripts/update_tickets_only.py"],
                                    cwd=project_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,  # Line buffered
                                    universal_newlines=True
                                )
                                
                                # Stream output in real-time
                                for line in iter(process.stdout.readline, ''):
                                    if line:
                                        line = line.strip()
                                        if line:
                                            status.write(line)
                                
                                process.wait()
                                
                                if process.returncode == 0:
                                    status.update(label="✅ Complete! Knowledge base updated", state="complete")
                                    # Refresh stats
                                    try:
                                        st.session_state.stats = st.session_state.pipeline.get_stats()
                                    except:
                                        pass
                                    unique_count = st.session_state.get("unique_tickets_count", len(unique_tickets))
                                    st.success(f"🎉 Successfully imported **{unique_count}** new tickets!")
                                    st.balloons()
                                    # Clear session state
                                    if "unique_tickets_to_process" in st.session_state:
                                        del st.session_state.unique_tickets_to_process
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    status.update(label="❌ Update failed", state="error")
                                    st.error("Failed to update knowledge base. Check logs above.")
                                    st.code(result.stdout)
                            except Exception as e:
                                status.write(f"❌ Error: {e}")
                                st.error(f"Update failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    
            except Exception as e:
                st.error(f"❌ Error checking knowledge base: {e}")
                import traceback
                with st.expander("🔍 Error Details", expanded=False):
                    st.code(traceback.format_exc())
                        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc())
    
    
    st.markdown("---")
    
    # Refresh Guides Section
    st.markdown("#### 🌐 **Refresh Guides**")
    st.markdown("*Scrape latest guides from website, chunk, and update embeddings*")
    
    if st.button("🔄 Refresh Guides", use_container_width=True):
        with st.status("Processing guides...", expanded=True) as status:
            try:
                # Step 1: Scrape guides
                status.write("📥 Step 1/4: Scraping guides from website...")
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
                
                if process.returncode != 0:
                    status.update(label="❌ Scraping failed", state="error")
                    st.error("Failed to scrape guides")
                    st.stop()
                
                status.write("✅ Guides scraped!")
                
                # Step 2: Check for new guides
                status.write("🔍 Step 2/4: Checking for new guides...")
                from config.settings import GUIDES_COMBINED_FILE, GUIDES_CHUNKS_FILE
                import json
                
                # Load existing guides (if any)
                existing_guide_urls = set()
                if GUIDES_COMBINED_FILE.exists():
                    try:
                        with open(GUIDES_COMBINED_FILE, 'r', encoding='utf-8') as f:
                            existing_guides = json.load(f)
                            existing_guide_urls = {g.get('url', '') for g in existing_guides if g.get('url')}
                    except:
                        pass
                
                # Load newly scraped guides
                new_guides = []
                if GUIDES_COMBINED_FILE.exists():
                    with open(GUIDES_COMBINED_FILE, 'r', encoding='utf-8') as f:
                        new_guides = json.load(f)
                
                new_guide_urls = {g.get('url', '') for g in new_guides if g.get('url')}
                new_count = len(new_guide_urls - existing_guide_urls)
                
                if new_count > 0:
                    status.write(f"✅ Found {new_count} new guide(s)!")
                else:
                    status.write("ℹ️  No new guides found (all already exist)")
                
                # Step 3: Chunk guides (always regenerate chunks to catch updates)
                status.write("✂️  Step 3/4: Chunking guides...")
                chunk_process = subprocess.run(
                    [sys.executable, "-m", "src.phase1.semantic_chunker"],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60
                )
                
                for line in chunk_process.stdout.split('\n'):
                    if line.strip():
                        status.write(line.strip())
                
                if chunk_process.returncode != 0:
                    status.update(label="❌ Chunking failed", state="error")
                    st.error("Failed to chunk guides")
                    st.code(chunk_process.stdout)
                    st.stop()
                
                status.write("✅ Guides chunked!")
                
                # Step 4: Update embeddings (incremental - only new chunks)
                status.write("🔍 Step 4/4: Updating embeddings (only new chunks)...")
                update_process = subprocess.run(
                    [sys.executable, "scripts/update_guides_incremental.py"],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300
                )
                
                for line in update_process.stdout.split('\n'):
                    if line.strip():
                        status.write(line.strip())
                
                if update_process.returncode == 0:
                    status.update(label="✅ Complete! Guides updated", state="complete")
                    # Refresh stats
                    try:
                        st.session_state.stats = st.session_state.pipeline.get_stats()
                    except:
                        pass
                    if new_count > 0:
                        st.success(f"🎉 Successfully processed **{new_count}** new guide(s)!")
                    else:
                        st.info("ℹ️  Guides refreshed (no new guides found)")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="❌ Update failed", state="error")
                    st.error("Failed to update guide embeddings")
                    st.code(update_process.stdout)
                    
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show last refresh timestamp
    from config.settings import GUIDES_CHUNKS_FILE
    if GUIDES_CHUNKS_FILE.exists():
        import os
        last_modified = datetime.fromtimestamp(os.path.getmtime(GUIDES_CHUNKS_FILE))
        st.caption(f"📅 Last refreshed: {last_modified.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.caption("📅 Never refreshed")
    
    st.markdown("---")
    
    # Full Rebuild (for emergencies)
    st.markdown("#### 🔨 **Full Rebuild**")
    st.markdown("*Complete database recreation (use only if needed)*")
    
    if st.button("🔨 Full Rebuild", use_container_width=True):
        with st.status("Rebuilding database...", expanded=True) as status:
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/rebuild_vector_db_v2.py"],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=600
                )
                
                for line in result.stdout.split('\n'):
                    if line.strip():
                        status.write(line.strip())
                
                if result.returncode == 0:
                    status.update(label="✅ Database rebuilt!", state="complete")
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="❌ Rebuild failed", state="error")
                    st.error(result.stdout)
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
