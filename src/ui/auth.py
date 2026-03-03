"""
Authentication for the Streamlit app. VPS-ready.
Uses streamlit-authenticator with 7-day cookie for session persistence across refresh.
"""

import os

import streamlit as st

_COOKIE_EXPIRY_DAYS = 7


def _build_credentials():
    """Build credentials dict from .env for streamlit-authenticator."""
    username = os.getenv("AUTH_USERNAME", "").strip()
    password_hash = os.getenv("AUTH_PASSWORD_HASH", "").strip()
    display_name = os.getenv("AUTH_DISPLAY_NAME", "").strip() or username
    if not username or not password_hash:
        return None
    return {
        "usernames": {
            username: {
                "email": f"{username}@local",
                "name": display_name,
                "password": password_hash,
            }
        }
    }


def _get_authenticator():
    """Create Authenticate instance (cookie expires in 7 days)."""
    import streamlit_authenticator as stauth

    creds = _build_credentials()
    if not creds:
        return None

    cookie_key = os.getenv("AUTH_COOKIE_SECRET", "").strip() or os.getenv("AUTH_PASSWORD_HASH", "lacura-signature-key")
    return stauth.Authenticate(
        credentials=creds,
        cookie_name="lacura_auth",
        cookie_key=cookie_key,
        cookie_expiry_days=_COOKIE_EXPIRY_DAYS,
        auto_hash=False,
    )


def is_auth_configured() -> bool:
    """Check if auth env vars are set."""
    return _build_credentials() is not None


def get_credentials():
    """Return (username, display_name, password_hash) or None if not configured."""
    creds = _build_credentials()
    if not creds:
        return None
    for uname, data in creds["usernames"].items():
        return (uname, data["name"], data["password"])
    return None


def require_authenticated_user():
    """
    Ensure user is authenticated. If not, show login and stop.
    Returns (display_name, username) when authenticated.
    Session persists 7 days via cookie across page refresh.
    """
    if not is_auth_configured():
        return ("User", "user")

    authenticator = _get_authenticator()
    if not authenticator:
        st.error("Authentication not configured. Set AUTH_USERNAME and AUTH_PASSWORD_HASH in .env")
        st.stop()

    # Inject dark theme for login page
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    .stApp, .stApp > div, html, body, [data-testid="stAppViewContainer"] {
        background: #05050a !important;
        background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.15), transparent) !important;
    }
    .login-header { text-align: center; margin-bottom: 1.5rem; }
    .login-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .login-title { font-family: 'Outfit', sans-serif; color: #f0f4ff; font-size: 1.75rem; font-weight: 700; }
    .login-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 0.25rem; }
    [data-testid="stTextInput"] input { background: #0c0c12 !important; color: #f0f4ff !important; border: 1px solid #2a2a3a !important; border-radius: 12px !important; }
    [data-testid="stFormSubmitButton"] button { background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important; color: white !important; border-radius: 12px !important; font-weight: 600 !important; }
    </style>
    <div class="login-header">
        <div class="login-icon">🚗</div>
        <div class="login-title">LaCuraDellAuto AI</div>
        <div class="login-subtitle">Sign in to access the support assistant</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        authenticator.login(
            location="main",
            fields={
                "Form name": "🔐 Secure Access",
                "Username": "Username",
                "Password": "Password",
                "Login": "Sign In",
            },
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.session_state.get("authentication_status"):
        return (
            st.session_state.get("name", "User"),
            st.session_state.get("username", "user"),
        )

    if st.session_state.get("authentication_status") is False:
        st.error("Invalid username or password.")
    st.stop()


def logout():
    """Clear auth session and cookie."""
    authenticator = _get_authenticator()
    if authenticator:
        try:
            authenticator.logout(location="unrendered")
        except Exception:
            pass
    for key in ("authentication_status", "name", "username", "logout"):
        st.session_state.pop(key, None)
