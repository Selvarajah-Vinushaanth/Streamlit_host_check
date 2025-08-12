import streamlit as st
from firebase_auth import FirebaseAuth
from firebase_config import FIREBASE_CONFIG
from datetime import datetime, timedelta
import time

def handle_google_callback():
    """Handle Google OAuth callback"""
    # Initialize Firebase Auth
    firebase_auth = FirebaseAuth(FIREBASE_CONFIG)
    
    # Get query parameters
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        code = query_params['code']
        state = query_params['state']
        
        # Get the callback URL
        callback_url = st.get_option("server.baseUrlPath") or ""
        if callback_url.endswith("/"):
            callback_url = callback_url[:-1]
            
        # Append current path
        callback_url += st.runtime.scriptrunner.get_script_run_ctx().page_script_hash
        
        # Verify state matches to prevent CSRF
        if 'oauth_state' in st.session_state and state == st.session_state.oauth_state:
            st.success("Google authentication successful! Processing...")
            
            # Exchange code for tokens
            result = firebase_auth.exchange_google_code(code, callback_url)
            
            if result['success']:
                # Update session state
                st.session_state.authenticated = True
                st.session_state.auth_token = result['token']
                st.session_state.refresh_token = result['refresh_token']
                st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
                st.session_state.user_info = {
                    'user_id': result['user_id'],
                    'email': result['email'],
                    'display_name': result.get('display_name', '')
                }
                
                st.success("✅ Successfully signed in with Google!")
                time.sleep(2)
                
                # Redirect to main page
                st.query_params.clear()  # Clear parameters
                st.rerun()
            else:
                st.error(f"❌ Authentication failed: {result['error']}")
        else:
            st.error("❌ Invalid authentication state. Please try again.")
    else:
        st.error("❌ Missing required parameters for Google authentication.")
    
    # Add a button to go back to login
    if st.button("← Return to Login"):
        st.query_params.clear()  # Clear parameters
        st.rerun()
