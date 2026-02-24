"""
Authentication for the Streamlit app. VPS-ready, session-based.
No cookies - avoids refresh/domain issues when hosted online.
"""

import os

import streamlit as st

def _verify_password(plain: str, hashed: str) -> bool:
    """Verify password against bcrypt hash (compatible with streamlit-authenticator format)."""
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def is_auth_configured() -> bool:
    """Check if auth env vars are set."""
    username = os.getenv("AUTH_USERNAME", "").strip()
    password_hash = os.getenv("AUTH_PASSWORD_HASH", "").strip()
    return bool(username and password_hash)


def get_credentials():
    """Return (username, display_name, password_hash) or None if not configured."""
    username = os.getenv("AUTH_USERNAME", "").strip()
    password_hash = os.getenv("AUTH_PASSWORD_HASH", "").strip()
    display_name = os.getenv("AUTH_DISPLAY_NAME", "").strip() or username
    if username and password_hash:
        return (username, display_name, password_hash)
    return None


def render_login_page():
    """Render login form. Returns (display_name, True) on success, else (None, False)."""
    creds = get_credentials()
    if not creds:
        st.error("Authentication not configured. Set AUTH_USERNAME and AUTH_PASSWORD_HASH in .env")
        st.stop()

    username, display_name, password_hash = creds
    prev_authenticated = st.session_state.get("auth_authenticated", False)

    st.markdown("""
    <style>
        .login-wrap { min-height: 75vh; display: flex; align-items: center; justify-content: center; }
        .login-card {
            max-width: 420px; margin: 0 auto;
            background: linear-gradient(135deg, #151525 0%, #1f1f33 100%);
            border: 1px solid #2f3152; border-radius: 16px;
            padding: 2rem; box-shadow: 0 12px 40px rgba(0,0,0,0.35);
        }
        .login-title { color: #e6e8ff; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }
        .login-subtitle { color: #aab1d6; font-size: 0.9rem; margin-bottom: 1.5rem; }
        .login-badge {
            display: inline-block; background: rgba(99,102,241,0.2); color: #bfc6ff;
            border: 1px solid rgba(99,102,241,0.45); border-radius: 999px;
            padding: 0.2rem 0.6rem; font-size: 0.75rem; margin-bottom: 1rem;
        }
        [data-testid="stTextInput"] input { background: #0f1326 !important; color: #f0f3ff !important; }
        [data-testid="stFormSubmitButton"] button {
            width: 100% !important; background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
            color: white !important; font-weight: 600 !important; border-radius: 10px !important;
        }
    </style>
    <div class="login-wrap">
        <div class="login-card">
            <div class="login-badge">Secure Access</div>
            <div class="login-title">LaCuraDellAuto AI</div>
            <div class="login-subtitle">Sign in to access the support assistant</div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        inp_username = st.text_input("Username", placeholder="Enter username", autocomplete="username")
        inp_password = st.text_input("Password", type="password", placeholder="Enter password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign In")

    st.markdown("</div></div>", unsafe_allow_html=True)

    if submitted and inp_username and inp_password:
        if inp_username.strip().lower() == username.lower() and _verify_password(inp_password, password_hash):
            st.session_state.auth_authenticated = True
            st.session_state.auth_username = username
            st.session_state.auth_display_name = display_name
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return (display_name if prev_authenticated else None, st.session_state.get("auth_authenticated", False))


def require_authenticated_user():
    """
    Ensure user is authenticated. If not, show login and stop.
    Returns (display_name, username) when authenticated.
    """
    if not is_auth_configured():
        return ("User", "user")  # No auth configured = open access

    if st.session_state.get("auth_authenticated"):
        return (
            st.session_state.get("auth_display_name", "User"),
            st.session_state.get("auth_username", "user"),
        )

    render_login_page()
    st.stop()


def logout():
    """Clear auth session state."""
    for key in ("auth_authenticated", "auth_username", "auth_display_name"):
        st.session_state.pop(key, None)
