"""Session and pipeline state helpers for Streamlit app."""

import time


def reset_pipeline(st, recycle_client: bool = False):
    """Clear active pipeline references and rebuild on next rerun."""
    if "pipeline" in st.session_state and st.session_state.pipeline is not None:
        try:
            st.session_state.pipeline.chroma = None
            st.session_state.pipeline.bm25_retriever = None
        except Exception:
            pass
    st.session_state.pipeline = None
    st.session_state.initialized = False
    if recycle_client and "_chroma_client" in st.session_state:
        try:
            st.session_state._chroma_client._system.stop()
        except Exception:
            pass
        st.session_state.pop("_chroma_client", None)
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        time.sleep(1)


def init_session_state(st):
    """Initialize all app-level session keys."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.pipeline = None
        st.session_state.current_model = None
        st.session_state.current_provider = None
        st.session_state.stats = {"tickets": 0, "guides": 0}
        st.session_state.query_history = []
        st.session_state.current_response = None
        st.session_state.current_context = None
    if "stats" not in st.session_state:
        st.session_state.stats = {"tickets": 0, "guides": 0}


def get_chroma_client(st):
    """Get a healthy ChromaDB client; recreate only when truly broken."""
    import gc
    import chromadb
    from config.settings import CHROMA_DB_DIR

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    existing_client = st.session_state.get("_chroma_client")
    if existing_client is not None:
        try:
            existing_client.heartbeat()
            return existing_client
        except Exception:
            st.session_state.pop("_chroma_client", None)
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass

    last_err = None
    for _ in range(5):
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            client.heartbeat()
            st.session_state._chroma_client = client
            return client
        except Exception as e:
            last_err = e
            gc.collect()
            time.sleep(2)

    raise RuntimeError(f"Could not connect to ChromaDB after 5 retries: {last_err}")
