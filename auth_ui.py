import streamlit as st
from firebase_auth import FirebaseAuth, init_session_state, logout, check_token_expiry
from firebase_config import FIREBASE_CONFIG, AUTH_CONFIG
import re
import time
from datetime import datetime, timedelta

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < AUTH_CONFIG['password_min_length']:
        return False, f"Password must be at least {AUTH_CONFIG['password_min_length']} characters long"
    return True, "Password is valid"

def render_login_form(firebase_auth):
    """Render login form"""
    st.markdown("### 🔐 Login to Your Account")
    
    # Get the callback URL for Google OAuth
    # Use current URL as the callback URL
    google_auth_callback_url = st.get_option("server.baseUrlPath") or ""
    if google_auth_callback_url.endswith("/"):
        google_auth_callback_url = google_auth_callback_url[:-1]
    
    # Append current path
    google_auth_callback_url += st.runtime.scriptrunner.get_script_run_ctx().page_script_hash
    
    # Google Sign-In Button
    google_auth_url = firebase_auth.get_google_auth_url(google_auth_callback_url)
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 15px;">
        <a href="{google_auth_url}" target="_self">
            <button style="
                background-color: white;
                color: #757575;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                display: inline-flex;
                align-items: center;
                cursor: pointer;
                width: 100%;
                justify-content: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                ">
                <img src="https://developers.google.com/identity/images/g-logo.png" 
                     style="height: 18px; margin-right: 10px;" alt="Google logo">
                Sign in with Google
            </button>
        </a>
    </div>
    <div style="text-align: center; margin: 15px 0;">
        <span style="color: #666; font-size: 14px;">- OR -</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form_unique", clear_on_submit=False):  # Updated key
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)
        
        with col2:
            forgot_password = st.form_submit_button("🔑 Forgot Password", use_container_width=True)
        
        with col3:
            show_signup = st.form_submit_button("📝 Create Account", use_container_width=True)
    
    if login_button:
        if not email or not password:
            st.error("❌ Please enter both email and password")
            return False
        
        if not validate_email(email):
            st.error("❌ Please enter a valid email address")
            return False
        
        with st.spinner("🔄 Signing you in..."):
            result = firebase_auth.sign_in(email, password)
            
            if result['success']:
                st.session_state.authenticated = True
                st.session_state.auth_token = result['token']
                st.session_state.refresh_token = result['refresh_token']
                st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
                st.session_state.user_info = {
                    'user_id': result['user_id'],
                    'email': result['email'],
                    'display_name': result.get('display_name', '')
                }
                st.success("✅ Login successful! Welcome back!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ Login failed: {result['error']}")
    
    if forgot_password:
        st.session_state.show_reset_form = True
        st.rerun()
    
    if show_signup:
        st.session_state.show_signup_form = True
        st.rerun()
    
    # Handle Google OAuth callback
    if 'code' in st.query_params:
        code = st.query_params['code']
        state = st.query_params.get('state', '')
        
        # Verify state to prevent CSRF
        if 'oauth_state' in st.session_state and state == st.session_state.oauth_state:
            with st.spinner("🔄 Signing in with Google..."):
                result = firebase_auth.exchange_google_code(code, google_auth_callback_url)
                
                if result['success']:
                    st.session_state.authenticated = True
                    st.session_state.auth_token = result['token']
                    st.session_state.refresh_token = result['refresh_token']
                    st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
                    st.session_state.user_info = {
                        'user_id': result['user_id'],
                        'email': result['email'],
                        'display_name': result.get('display_name', '')
                    }
                    
                    # Clear the URL parameters
                    time.sleep(1)
                    st.success("✅ Google Sign-in successful!")
                    time.sleep(1)
                    
                    # Redirect to clean URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Google Sign-in failed: {result['error']}")
        else:
            st.error("❌ Invalid authentication state. Please try again.")
            # Clear invalid parameters
            st.query_params.clear()
    
    return False

def render_signup_form(firebase_auth):
    """Render signup form"""
    st.markdown("### 📝 Create New Account")
    
    with st.form("signup_form_unique", clear_on_submit=False):  # Updated key
        display_name = st.text_input("👤 Full Name", placeholder="Enter your full name")
        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            signup_button = st.form_submit_button("🚀 Create Account", use_container_width=True)
        
        with col2:
            back_to_login = st.form_submit_button("🔙 Back to Login", use_container_width=True)
    
    if signup_button:
        # Validation
        if not all([display_name, email, password, confirm_password]):
            st.error("❌ Please fill in all fields")
            return False
        
        if not validate_email(email):
            st.error("❌ Please enter a valid email address")
            return False
        
        is_valid, message = validate_password(password)
        if not is_valid:
            st.error(f"❌ {message}")
            return False
        
        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return False
        
        with st.spinner("🔄 Creating your account..."):
            result = firebase_auth.sign_up(email, password, display_name)
            
            if result['success']:
                st.success("✅ Account created successfully! Please login with your credentials.")
                st.session_state.show_signup_form = False
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ Account creation failed: {result['error']}")
    
    if back_to_login:
        st.session_state.show_signup_form = False
        st.rerun()
    
    return False

def render_reset_form(firebase_auth):
    """Render password reset form"""
    st.markdown("### 🔑 Reset Your Password")
    
    with st.form("reset_form_unique", clear_on_submit=False):  # Updated key
        email = st.text_input("📧 Email", placeholder="Enter your email address")
        
        col1, col2 = st.columns(2)
        
        with col1:
            reset_button = st.form_submit_button("📤 Send Reset Email", use_container_width=True)
        
        with col2:
            back_to_login = st.form_submit_button("🔙 Back to Login", use_container_width=True)
    
    if reset_button:
        if not email:
            st.error("❌ Please enter your email address")
            return False
        
        if not validate_email(email):
            st.error("❌ Please enter a valid email address")
            return False
        
        with st.spinner("📤 Sending reset email..."):
            result = firebase_auth.reset_password(email)
            
            if result['success']:
                st.success("✅ Password reset email sent! Check your inbox.")
                st.session_state.show_reset_form = False
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ Failed to send reset email: {result['error']}")
    
    if back_to_login:
        st.session_state.show_reset_form = False
        st.rerun()
    
    return False

def render_user_menu():
    """Render user menu in sidebar"""
    if st.session_state.get('authenticated') and st.session_state.get('user_info'):
        user_info = st.session_state.user_info
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Profile")
        
        # User info display
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <div style="font-weight: bold; font-size: 1.1rem;">👋 Welcome!</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">{user_info.get('display_name', 'User')}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">{user_info.get('email', '')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # User actions
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("👤 Profile", use_container_width=True):
                st.session_state.show_profile = True
        
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                logout()

def render_authentication_page():
    """Main authentication page"""
    # Initialize session state
    init_session_state()
    
    try:
        # Initialize Firebase Auth
        firebase_auth = FirebaseAuth(FIREBASE_CONFIG)
        
        # Check token expiry
        if st.session_state.get('authenticated'):
            check_token_expiry(firebase_auth)
        
        # If user is authenticated, return True
        if st.session_state.get('authenticated'):
            render_user_menu()
            return True
        
        # Authentication UI
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; color: white;">
            <h1 style="margin: 0; font-size: 2.5rem;">🔐 Authentication</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Secure access to Tamil Metaphor Classifier</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show appropriate form
        if st.session_state.get('show_signup_form'):
            render_signup_form(firebase_auth)
        elif st.session_state.get('show_reset_form'):
            render_reset_form(firebase_auth)
        else:
            render_login_form(firebase_auth)
        
        return False
        
    except Exception as e:
        st.error(f"❌ Authentication system error: {str(e)}")
        st.info("Please check your Firebase configuration and internet connection.")
        return False

# Initialize session state variables for forms
if 'show_signup_form' not in st.session_state:
    st.session_state.show_signup_form = False
if 'show_reset_form' not in st.session_state:
    st.session_state.show_reset_form = False
if 'show_profile' not in st.session_state:
    st.session_state.show_profile = False
