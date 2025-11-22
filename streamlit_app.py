"""
LaCuraDellAuto AI Support Assistant - Streamlit Interface
A modern, professional interface for customer support agents.
"""

import streamlit as st
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import streamlit.components.v1 as components

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase4.rag_pipeline import RAGPipeline
from src.phase4.vector_db import VectorDBManager
from src.utils.model_checker import get_available_models, is_model_available

# Initialize copy state
if 'copy_trigger' not in st.session_state:
    st.session_state.copy_trigger = None
    st.session_state.copy_text = None


# Page configuration
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional look
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1e40af;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #3b82f6;
        font-weight: 600;
    }
    
    /* Query input area */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Response container */
    .response-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Context boxes */
    .context-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Stats */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3b82f6;
    }
    
    /* History item */
    .history-item {
        background: white;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid #d1d5db;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        border-left-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Info messages */
    .success-msg {
        background: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f9fafb;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pipeline = None
    st.session_state.history = []
    st.session_state.current_response = None
    st.session_state.current_context = None
    st.session_state.loading = False
    st.session_state.stats = None

# Initialize RAG Pipeline (cached)
@st.cache_resource
def initialize_pipeline(model_name):
    """Initialize the RAG pipeline (runs once per model)."""
    try:
        import os
        # Force the model to be used
        os.environ['OLLAMA_MODEL'] = model_name
        
        # Clear any cached imports
        if 'config.settings' in sys.modules:
            del sys.modules['config.settings']
        
        pipeline = RAGPipeline(model=model_name)
        return pipeline, None
    except Exception as e:
        return None, str(e)

# Model selection (optimized for speed + quality)
# Preferred models in order of preference
PREFERRED_MODELS = [
    'mistral:7b-instruct',           # RECOMMENDED: Fast + Best quality (7B quantized) ⚡
    'mixtral:8x7b-instruct',         # Maximum quality (8x7B quantized) - if available
    'qwen2.5:7b-instruct',           # Balanced (7B quantized)
    'llama3.1:8b',                   # Alternative (8B quantized)
    'gemma2:2b',                     # Fast baseline (2B - fast but lower quality)
    'qwen2.5:14b',                   # High quality (14B - slower)
]

# Get actually available models
try:
    installed_models = get_available_models()
    
    # Build available models list by matching preferred to installed
    AVAILABLE_MODELS = []
    used_models = set()
    
    # First, try to match preferred models to installed ones
    for preferred in PREFERRED_MODELS:
        base_name = preferred.split(':')[0]  # e.g., 'mistral' from 'mistral:7b-instruct'
        
        # Find matching installed models
        for installed in installed_models:
            if installed not in used_models:
                # Check if base name matches (e.g., 'mistral' matches 'mistral:7b-instruct')
                if installed.startswith(base_name + ':') or installed == base_name:
                    AVAILABLE_MODELS.append(installed)
                    used_models.add(installed)
                    break  # Use first match for this preferred model
    
    # Add any remaining installed models not in preferred list
    for installed in installed_models:
        if installed not in used_models:
            AVAILABLE_MODELS.append(installed)
    
    # Ensure we have at least gemma2:2b if nothing else works
    if not AVAILABLE_MODELS:
        AVAILABLE_MODELS = installed_models if installed_models else ['gemma2:2b']
    
    # Default to mistral:7b-instruct if available, otherwise gemma2:2b
    default_index = 0
    if 'mistral:7b-instruct' in AVAILABLE_MODELS:
        default_index = AVAILABLE_MODELS.index('mistral:7b-instruct')
    elif 'gemma2:2b' in AVAILABLE_MODELS:
        default_index = AVAILABLE_MODELS.index('gemma2:2b')
        
except Exception as e:
    # Fallback to known models if check fails
    st.warning(f"Error checking models: {e}")
    AVAILABLE_MODELS = ['mistral:7b-instruct', 'gemma2:2b', 'qwen2.5:7b-instruct', 'llama3.1:8b', 'qwen2.5:14b']
    default_index = 0

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    AVAILABLE_MODELS,
    index=default_index,
    help="Select an Ollama model. If model not available, pull it first: ollama pull <model-name>"
)

# Check if model changed
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

model_changed = st.session_state.current_model != selected_model

# Load pipeline on first run or model change
if not st.session_state.initialized or model_changed:
    with st.spinner(f"🚀 Initializing AI Assistant with {selected_model}..."):
        # Clear cache if model changed
        if model_changed and st.session_state.initialized:
            st.cache_resource.clear()
        
        pipeline, error = initialize_pipeline(selected_model)
        if error:
            st.error(f"❌ Failed to initialize model '{selected_model}': {error}")
            if "not found" in str(error).lower() or "404" in str(error):
                st.warning(f"💡 Model '{selected_model}' is not installed. Pull it first:")
                st.code(f"ollama pull {selected_model}", language="bash")
                st.info("After pulling, refresh this page.")
            st.stop()
        else:
            st.session_state.pipeline = pipeline
            st.session_state.initialized = True
            st.session_state.current_model = selected_model
            
            # Get initial stats
            try:
                st.session_state.stats = pipeline.db_manager.get_stats()
            except:
                st.session_state.stats = {'tickets': 0, 'guides': 0}

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Retrieval settings
    st.markdown("### 🔍 Retrieval Settings")
    n_tickets = st.slider("Number of tickets", 1, 5, 3)
    n_guides = st.slider("Number of guides", 1, 5, 3)
    
    st.divider()
    
    # Draft generation settings
    st.markdown("### 📝 Draft Options")
    num_drafts = st.selectbox(
        "Number of drafts",
        options=[1, 2, 3],
        index=0,  # Default to 1 draft (optimized for speed)
        help="Generate multiple draft responses. Note: Sequential generation (optimized for single GPU). For fastest responses, use 1 draft."
    )
    
    if num_drafts > 1:
        st.caption(f"⚡ Will generate {num_drafts} variations sequentially (optimized for single GPU)")
        # Timing estimates based on selected model
        if selected_model == 'gemma2:2b':
            # Fast 2B model
            time_estimates = {1: 8, 2: 15, 3: 22}
        elif '7b' in selected_model or '8b' in selected_model:
            # Medium 7-8B models
            time_estimates = {1: 30, 2: 55, 3: 75}
        else:
            # Larger models (14B, etc.)
            time_estimates = {1: 60, 2: 110, 3: 150}
        
        est_time = time_estimates.get(num_drafts, num_drafts * (time_estimates.get(1, 30)))
        st.caption(f"⏱️ Est. time: ~{est_time}s")
    
    # Statistics
    st.markdown("### 📊 Database Stats")
    if st.session_state.stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tickets", st.session_state.stats['tickets'])
        with col2:
            st.metric("Guides", st.session_state.stats['guides'])
    
    st.divider()
    
    # History
    st.markdown("### 📜 Query History")
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} queries")
        
        # Show last 5 queries
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            # Format time display
            elapsed = item.get('time', 0)
            cached = item.get('cached', False)
            if cached:
                time_display = "⚡ <0.01s"
            else:
                time_display = f"⏱️ {elapsed:.2f}s"
            
            with st.expander(f"🕐 {item['timestamp']} | {time_display}", expanded=False):
                st.caption(item['query'][:100] + "..." if len(item['query']) > 100 else item['query'])
                if st.button(f"Load", key=f"load_{len(st.session_state.history)-i-1}"):
                    st.session_state.current_response = item['response']
                    st.session_state.current_context = item['context']
                    st.session_state.response_time = item.get('time', 0)
                    st.session_state.was_cached = item.get('cached', False)
                    st.rerun()
    else:
        st.caption("No queries yet")
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.current_response = None
        st.session_state.current_context = None
        st.rerun()
    
    # Clear cache
    if st.button("🔄 Clear Cache"):
        if hasattr(st.session_state.pipeline, '_cache'):
            cache_size = len(st.session_state.pipeline._cache)
            st.session_state.pipeline._cache.clear()
            st.success(f"✓ Cleared {cache_size} cached responses")
            st.rerun()
    
    # Performance info
    st.divider()
    st.markdown("### ⚡ Performance")
    
    # Calculate average response time from history
    if st.session_state.history:
        times = [h.get('time', 0) for h in st.session_state.history if not h.get('cached', False)]
        cached_count = sum(1 for h in st.session_state.history if h.get('cached', False))
        if times:
            avg_time = sum(times) / len(times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s", help="Average time for non-cached responses")
        if cached_count > 0:
            st.metric("Cache Hits", f"{cached_count}/{len(st.session_state.history)}", 
                     help="Number of instant cached responses")
    
    st.caption("✅ Italian-optimized prompt")
    st.caption("✅ Smart caching enabled")
    st.caption("✅ qwen2.5:7b-instruct")
    if hasattr(st.session_state.pipeline, '_cache'):
        cache_size = len(st.session_state.pipeline._cache)
        st.caption(f"📦 {cache_size} cached responses")

# Main interface
st.title("🤖 AI Support Assistant")
st.markdown("### LaCuraDellAuto - Customer Support Helper")

# Instructions
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    1. **Paste** the customer's question in Italian or English
    2. Click **Generate Response** 
    3. Review the AI-generated response and retrieved context
    4. **Copy** the response or **edit** it before sending to customer
    5. Optionally **rate** the response to help improve the system
    """)

# Query input area
st.markdown("#### 📝 Customer Query")
query = st.text_area(
    "Enter the customer's question:",
    height=120,
    placeholder="Example: Come posso rimuovere i graffi dalla mia auto?\n\nThe AI will search through historical tickets and technical guides to generate a helpful response.",
    label_visibility="collapsed"
)

# Action buttons
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    generate_btn = st.button("✨ Generate Response", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🔄 Clear", use_container_width=True)
with col3:
    example_btn = st.button("💡 Example", use_container_width=True)

# Handle example button
if example_btn:
    query = "Come posso lucidare la mia auto per rimuovere i graffi?"
    st.rerun()

# Handle clear button
if clear_btn:
    st.session_state.current_response = None
    st.session_state.current_context = None
    st.rerun()

# Handle generate button
if generate_btn and query.strip():
    # Timing estimates based on selected model
    if selected_model == 'gemma2:2b':
        # Fast 2B model - TARGET: ~20s for 3 drafts
        time_estimates = {1: 8, 2: 15, 3: 22}
    elif '7b' in selected_model or '8b' in selected_model:
        # Medium 7-8B models
        time_estimates = {1: 30, 2: 55, 3: 75}
    else:
        # Larger models (14B, etc.)
        time_estimates = {1: 60, 2: 110, 3: 150}
    
    est_time = time_estimates.get(num_drafts, num_drafts * (time_estimates.get(1, 30)))
    
    with st.spinner(f"🤔 Thinking... This may take ~{est_time} seconds"):
        try:
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Retrieving context
            status_text.text("🔍 Searching knowledge base...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Step 2: Generating response
            if num_drafts > 1:
                status_text.text(f"✨ Generating {num_drafts} drafts sequentially (optimized for GPU efficiency)...")
            else:
                status_text.text("✨ Generating response...")
            progress_bar.progress(50)
            
            # Track timing
            start_time = time.time()
            
            # Actual query
            result = st.session_state.pipeline.query(
                query,
                n_tickets=n_tickets,
                n_guides=n_guides,
                num_drafts=num_drafts
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            progress_bar.progress(90)
            status_text.text("✅ Complete!")
            time.sleep(0.3)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Determine if cached
            was_cached = elapsed_time < 1.0  # If response was instant, it was cached
            
            # Store results with timing
            st.session_state.current_response = result['response']
            st.session_state.current_responses = result.get('responses', None)  # Multiple drafts
            st.session_state.num_drafts = result.get('num_drafts', 1)
            st.session_state.current_context = result['context']
            st.session_state.response_time = elapsed_time
            st.session_state.was_cached = was_cached
            st.session_state.selected_draft = 0  # Default to first draft
            
            # Add to history
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'query': query,
                'response': result['response'],
                'context': result['context'],
                'time': elapsed_time,
                'cached': was_cached
            })
            
            # Success message with timing
            if was_cached:
                st.success(f"✅ Response generated instantly! (Cached)")
            elif num_drafts > 1:
                st.success(f"✅ {num_drafts} drafts generated in {elapsed_time:.2f} seconds! Choose your favorite below.")
            else:
                st.success(f"✅ Response generated in {elapsed_time:.2f} seconds!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display response(s)
if st.session_state.current_response:
    st.markdown("---")
    
    # Check if multiple drafts
    has_multiple_drafts = (st.session_state.get('current_responses') is not None and 
                           st.session_state.get('num_drafts', 1) > 1)
    
    if has_multiple_drafts:
        # Multiple drafts - show in tabs
        st.markdown("#### ✨ AI Generated Drafts - Choose Your Favorite")
        
        # Show timing
        if hasattr(st.session_state, 'response_time'):
            elapsed = st.session_state.response_time
            st.caption(f"⏱️ Generated {st.session_state.num_drafts} drafts in {elapsed:.2f} seconds")
        
        # Create tabs for each draft
        draft_responses = st.session_state.current_responses
        tab_labels = [f"Draft {i+1}" for i in range(len(draft_responses))]
        tabs = st.tabs(tab_labels)
        
        for i, tab in enumerate(tabs):
            with tab:
                draft = draft_responses[i]
                
                # Draft info
                col1, col2 = st.columns([3, 1])
                with col1:
                    if draft['temperature'] == 0.5:
                        st.caption("📘 Conservative (Most factual)")
                    elif draft['temperature'] == 0.7:
                        st.caption("📗 Balanced (Recommended)")
                    else:
                        st.caption("📙 Creative (More varied)")
                with col2:
                    st.caption(f"Temp: {draft['temperature']}")
                
                # Response text
                st.markdown(f"""
                <div class="response-box">
                    {draft['text'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons for this draft
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if st.button(f"✅ Select Draft {i+1}", key=f"select_draft_{i}", use_container_width=True):
                        st.session_state.selected_draft = i
                        st.session_state.current_response = draft['text']
                        st.success(f"✅ Draft {i+1} selected!")
                        st.rerun()
                with col2:
                    copy_clicked = st.button("📋 Copy", key=f"copy_draft_{i}", use_container_width=True)
                    if copy_clicked:
                        # Store text in session state and inject JavaScript to copy
                        st.session_state[f'copy_text_{i}'] = draft['text']
                        # Use JSON encoding for safe text handling
                        json_text = json.dumps(draft['text'])
                        copy_script = f"""
                        <script>
                        (function() {{
                            const text = JSON.parse({json_text});
                            const textarea = document.createElement('textarea');
                            textarea.value = text;
                            textarea.style.position = 'fixed';
                            textarea.style.left = '-999999px';
                            textarea.style.top = '-999999px';
                            document.body.appendChild(textarea);
                            textarea.focus();
                            textarea.select();
                            try {{
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text).then(function() {{
                                        document.body.removeChild(textarea);
                                    }}).catch(function() {{
                                        document.execCommand('copy');
                                        document.body.removeChild(textarea);
                                    }});
                                }} else {{
                                    document.execCommand('copy');
                                    document.body.removeChild(textarea);
                                }}
                            }} catch (err) {{
                                document.body.removeChild(textarea);
                            }}
                        }})();
                        </script>
                        """
                        components.html(copy_script, height=0)
                        st.success("✅ Copied to clipboard!")
                with col3:
                    # Show if this is currently selected
                    if st.session_state.get('selected_draft', 0) == i:
                        st.success("✓ Selected")
        
        st.markdown("---")
        st.caption("💡 Tip: Different drafts use varying creativity levels. Draft 1 is most factual, Draft 3 is most creative.")
        
    else:
        # Single response - original display
        # Response header with timing
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### ✨ AI Generated Response")
        with col2:
            if hasattr(st.session_state, 'response_time'):
                elapsed = st.session_state.response_time
                if st.session_state.get('was_cached', False):
                    st.markdown(f"<div style='text-align: right; color: #10b981; font-size: 0.9em;'>⚡ <b>Cached</b> (<0.01s)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: right; color: #6366f1; font-size: 0.9em;'>⏱️ <b>{elapsed:.2f}s</b></div>", unsafe_allow_html=True)
        
        # Response container
        st.markdown(f"""
        <div class="response-box">
            {st.session_state.current_response.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    
    # Action buttons for response (only for single response mode)
    if not has_multiple_drafts:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            copy_clicked = st.button("📋 Copy Response", use_container_width=True)
            if copy_clicked:
                # Store text in session state and inject JavaScript to copy
                st.session_state['copy_text_main'] = st.session_state.current_response
                # Use JSON encoding for safe text handling
                json_text = json.dumps(st.session_state.current_response)
                copy_script = f"""
                <script>
                (function() {{
                    const text = JSON.parse({json_text});
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.left = '-999999px';
                    textarea.style.top = '-999999px';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    try {{
                        if (navigator.clipboard && navigator.clipboard.writeText) {{
                            navigator.clipboard.writeText(text).then(function() {{
                                document.body.removeChild(textarea);
                            }}).catch(function() {{
                                document.execCommand('copy');
                                document.body.removeChild(textarea);
                            }});
                        }} else {{
                            document.execCommand('copy');
                            document.body.removeChild(textarea);
                        }}
                    }} catch (err) {{
                        document.body.removeChild(textarea);
                    }}
                }})();
                </script>
                """
                components.html(copy_script, height=0)
                st.success("✅ Copied to clipboard!")
        
        with col2:
            if st.button("👍 Good", use_container_width=True):
                st.success("Thanks for the feedback!")
        
        with col3:
            if st.button("👎 Bad", use_container_width=True):
                st.warning("Thanks! We'll improve.")
    
    # Editable version
    with st.expander("✏️ Edit Response", expanded=False):
        edited_response = st.text_area(
            "Edit the response before sending:",
            value=st.session_state.current_response,
            height=200,
            key="edit_area"
        )
        if st.button("💾 Save Edited Version"):
            st.session_state.current_response = edited_response
            st.success("✅ Response updated!")
            st.rerun()

# Display context
if st.session_state.current_context:
    st.markdown("---")
    
    with st.expander("🎯 Retrieved Context (Sources)", expanded=True):
        tabs = st.tabs(["📋 Tickets", "📚 Guides"])
        
        # Tickets tab
        with tabs[0]:
            tickets = st.session_state.current_context['tickets']
            if tickets['ids'] and tickets['ids'][0]:
                st.markdown(f"**Found {len(tickets['ids'][0])} relevant tickets:**")
                for i, (doc, meta) in enumerate(zip(tickets['documents'][0], tickets['metadatas'][0]), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="context-box">
                            <strong>📋 Ticket {i}</strong><br>
                            <em>Subject:</em> {meta.get('subject', 'N/A')}<br>
                            <em>Status:</em> {meta.get('status', 'N/A')}<br>
                            <details>
                                <summary>View content</summary>
                                <p style="margin-top: 0.5rem; color: #6b7280;">{doc[:300]}...</p>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No relevant tickets found")
        
        # Guides tab
        with tabs[1]:
            guides = st.session_state.current_context['guides']
            if guides['ids'] and guides['ids'][0]:
                st.markdown(f"**Found {len(guides['ids'][0])} relevant guide sections:**")
                for i, (doc, meta) in enumerate(zip(guides['documents'][0], guides['metadatas'][0]), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="context-box">
                            <strong>📚 Guide {i}</strong><br>
                            <em>Guide:</em> {meta.get('guide_title', 'N/A')}<br>
                            <em>Section:</em> {meta.get('section_title', 'N/A')}<br>
                            <em>URL:</em> <a href="{meta.get('url', '#')}" target="_blank">View online</a><br>
                            <details>
                                <summary>View content</summary>
                                <p style="margin-top: 0.5rem; color: #6b7280;">{doc[:400]}...</p>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No relevant guides found")

# Footer
st.markdown("---")
st.caption(f"🤖 Powered by {st.session_state.pipeline.model if st.session_state.pipeline else 'AI'} | "
           f"💾 {len(st.session_state.history)} queries in history | "
           f"⚡ Ready to assist")

