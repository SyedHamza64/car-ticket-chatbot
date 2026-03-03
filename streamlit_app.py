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
from dotenv import load_dotenv

from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from src.app.state import init_session_state, reset_pipeline, get_chroma_client
from src.ui.auth import require_authenticated_user, logout, is_auth_configured
from src.ui.query_tab import render_query_tab
from src.ui.manage_tab import render_manage_tab

# Page configuration
st.set_page_config(
    page_title="LaCuraDellAuto AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication (optional: if AUTH_USERNAME + AUTH_PASSWORD_HASH set in .env)
display_name, username = require_authenticated_user()

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

def _reset_pipeline(recycle_client: bool = False):
    """Wrapper around app state helper for backward compatibility."""
    reset_pipeline(st, recycle_client=recycle_client)


init_session_state(st)

# Loading placeholder - shows while pipeline init runs in sidebar (avoids flash of login/empty state)
loading_placeholder = st.empty()

def _get_chroma_client():
    """Wrapper around app state helper for backward compatibility."""
    return get_chroma_client(st)

# Initialize RAG Pipeline
def initialize_pipeline(model_name, provider="ollama"):
    """Initialize the RAG pipeline with a fresh ChromaDB client."""
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
        
        client = _get_chroma_client()
        from src.phase4.rag_pipeline_langchain import LangchainRAG
        pipeline = LangchainRAG(
            provider=provider,
            model=model_name,
            chroma_client=client,
        )

        return pipeline, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# SIDEBAR
# ============================================================================
with loading_placeholder.container():
    st.markdown(
        """
        <style>
        @keyframes loader-orbital {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes loader-inner {
            0%, 100% { transform: scale(0.8); opacity: 0.6; }
            50% { transform: scale(1); opacity: 1; }
        }
        @keyframes loader-dots {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        @keyframes loader-road {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        .load-wrap {
            text-align: center;
            padding: 4rem 2rem;
        }
        .load-orbital {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            position: relative;
        }
        .load-ring {
            position: absolute;
            inset: 0;
            border: 2px solid transparent;
            border-top-color: #6366f1;
            border-right-color: #8b5cf6;
            border-radius: 50%;
            animation: loader-orbital 1s linear infinite;
        }
        .load-ring:nth-child(2) {
            inset: 8px;
            border-top-color: #8b5cf6;
            border-right-color: #6366f1;
            animation-duration: 1.2s;
            animation-direction: reverse;
        }
        .load-ring:nth-child(3) {
            inset: 16px;
            border-top-color: #a78bfa;
            animation-duration: 0.8s;
        }
        .load-center {
            position: absolute;
            inset: 28px;
            background: radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 70%);
            border-radius: 50%;
            animation: loader-inner 1.5s ease-in-out infinite;
        }
        .load-dots {
            display: flex;
            justify-content: center;
            gap: 6px;
            margin-bottom: 1rem;
        }
        .load-dots span {
            width: 8px;
            height: 8px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 50%;
            animation: loader-dots 1.4s ease-in-out infinite both;
        }
        .load-dots span:nth-child(1) { animation-delay: 0s; }
        .load-dots span:nth-child(2) { animation-delay: 0.2s; }
        .load-dots span:nth-child(3) { animation-delay: 0.4s; }
        .load-road {
            height: 4px;
            background: rgba(99,102,241,0.2);
            border-radius: 4px;
            max-width: 200px;
            margin: 0 auto 1rem;
            overflow: hidden;
        }
        .load-road-inner {
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border-radius: 4px;
            animation: loader-road 2s ease-in-out infinite;
        }
        .load-text { font-size: 1.1rem; color: var(--text-secondary); margin-bottom: 0.25rem; font-weight: 500; }
        .load-sub { font-size: 0.9rem; color: var(--text-muted); }
        </style>
        <div class="load-wrap">
            <div class="load-orbital">
                <div class="load-ring"></div>
                <div class="load-ring"></div>
                <div class="load-ring"></div>
                <div class="load-center"></div>
            </div>
            <div class="load-dots"><span></span><span></span><span></span></div>
            <div class="load-road"><div class="load-road-inner"></div></div>
            <p class="load-text">Loading your assistant...</p>
            <p class="load-sub">Initializing AI pipeline · Please wait</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    if is_auth_configured():
        st.markdown(f"👤 **{display_name}**")
        if st.button("🚪 Logout", key="logout_btn"):
            logout()
            st.rerun()
    st.markdown("---")
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
        ["⚡ Groq (Cloud)", "✨ Gemini (Cloud)"],
        index=0,
        help="Select your AI provider"
    )
    # Map to backend provider name
    provider_name = "grok" if "Groq" in provider else "gemini"
    
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
        try:
            from src.utils.model_checker import get_available_models
            _models = get_available_models()
            _ollama_models = _models if _models else ['gemma2:2b', 'llama3.1:8b']
        except Exception:
            _ollama_models = ['gemma2:2b', 'llama3.1:8b']
        selected_model = st.selectbox(
            "🤖 **Ollama Model**",
            _ollama_models,
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
                _reset_pipeline()
            
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
            _reset_pipeline()
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
loading_placeholder.empty()  # Clear loading state, show main UI

# Force sidebar to always be visible with CSS
st.markdown("""
<style>
    /* Force sidebar to ALWAYS be visible and expanded */
    section[data-testid="stSidebar"] {
        min-width: 21rem !important;
        width: 21rem !important;
        transform: translateX(0) !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* Hide the collapse button since we want sidebar always visible */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    button.st-emotion-cache-17zm0w6,
    button.etdmgzm15 {
        display: none !important;
    }
    
    /* Ensure main content doesn't overlap sidebar */
    .main .block-container {
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    render_query_tab(
        st=st,
        datetime=datetime,
        time=time,
        initialize_pipeline=initialize_pipeline,
        reset_pipeline=_reset_pipeline,
        selected_model=selected_model,
        provider_name=provider_name,
    )

# ============================================================================
# TAB 2: KNOWLEDGE BASE MANAGEMENT
# ============================================================================
with tab_manage:
    render_manage_tab(
        st=st,
        project_root=project_root,
        sys=sys,
        subprocess=subprocess,
        Path=Path,
        json=json,
        time=time,
        datetime=datetime,
        reset_pipeline=_reset_pipeline,
    )

# Footer
st.markdown("---")
provider_name = st.session_state.get('current_provider', 'ollama')
provider_label = "Ollama" if provider_name == "ollama" else "Groq"
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 1rem 0;">
    Built with ❤️ for LaCuraDellAuto • Powered by {provider_label} ({st.session_state.current_model or 'AI'})
</div>
""", unsafe_allow_html=True)
