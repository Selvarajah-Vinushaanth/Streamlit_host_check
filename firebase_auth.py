import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import json
import requests
from datetime import datetime, timedelta
import time
import secrets
import urllib.parse

class FirebaseAuth:
    def __init__(self, config):
        self.config = config
        self.api_key = config['apiKey']
        self.auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts"
        
        # Initialize Firebase Admin SDK if not already initialized
        if not firebase_admin._apps:
            try:
                # You'll need to add your service account key file
                cred = credentials.Certificate("song-writing-assistant-4cd39-firebase-adminsdk-fbsvc-9527d47b41.json")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                st.error(f"Firebase initialization error: {e}")
    
    def sign_up(self, email, password, display_name=None):
        """Create a new user account"""
        try:
            url = f"{self.auth_url}:signUp?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            if display_name:
                payload["displayName"] = display_name
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'user_id': data['localId'],
                    'email': data['email'],
                    'token': data['idToken'],
                    'refresh_token': data['refreshToken']
                }
            else:
                return {
                    'success': False,
                    'error': data.get('error', {}).get('message', 'Unknown error')
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def sign_in(self, email, password):
        """Sign in an existing user"""
        try:
            url = f"{self.auth_url}:signInWithPassword?key={self.api_key}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'user_id': data['localId'],
                    'email': data['email'],
                    'token': data['idToken'],
                    'refresh_token': data['refreshToken'],
                    'display_name': data.get('displayName', '')
                }
            else:
                return {
                    'success': False,
                    'error': data.get('error', {}).get('message', 'Unknown error')
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def refresh_token(self, refresh_token):
        """Refresh the authentication token"""
        try:
            url = f"https://securetoken.googleapis.com/v1/token?key={self.api_key}"
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'token': data['id_token'],
                    'refresh_token': data['refresh_token']
                }
            else:
                return {'success': False, 'error': 'Token refresh failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def verify_token(self, token):
        """Verify if the token is valid"""
        try:
            decoded_token = auth.verify_id_token(token)
            return {'success': True, 'user_id': decoded_token['uid']}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def reset_password(self, email):
        """Send password reset email"""
        try:
            url = f"{self.auth_url}:sendOobCode?key={self.api_key}"
            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return {'success': True, 'message': 'Password reset email sent'}
            else:
                data = response.json()
                return {
                    'success': False,
                    'error': data.get('error', {}).get('message', 'Unknown error')
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_info(self, token):
        """Get user information"""
        try:
            url = f"{self.auth_url}:lookup?key={self.api_key}"
            payload = {"idToken": token}
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if response.status_code == 200:
                user_data = data['users'][0]
                return {
                    'success': True,
                    'user_id': user_data['localId'],
                    'email': user_data['email'],
                    'display_name': user_data.get('displayName', ''),
                    'email_verified': user_data.get('emailVerified', False),
                    'created_at': user_data.get('createdAt', ''),
                    'last_login': user_data.get('lastLoginAt', '')
                }
            else:
                return {'success': False, 'error': 'Failed to get user info'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_google_auth_url(self, redirect_uri):
        """Generate Google OAuth URL for sign-in"""
        # Generate a state token to prevent request forgery
        state = secrets.token_hex(16)
        st.session_state.oauth_state = state
        
        # Google OAuth2 endpoint
        base_url = "https://accounts.google.com/o/oauth2/v2/auth"
        
        # Parameters for Google authentication
        params = {
            "client_id": self.config.get('googleClientId', ''),
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "email profile",
            "state": state,
            "prompt": "select_account",
            "access_type": "offline"
        }
        
        # Build the URL
        auth_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        return auth_url
    
    def exchange_google_code(self, code, redirect_uri):
        """Exchange Google OAuth code for Firebase token"""
        try:
            # First, exchange code for Google token
            token_url = "https://oauth2.googleapis.com/token"
            token_payload = {
                "code": code,
                "client_id": self.config.get('googleClientId', ''),
                "client_secret": self.config.get('googleClientSecret', ''),
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code"
            }
            
            token_response = requests.post(token_url, data=token_payload)
            token_data = token_response.json()
            
            if 'error' in token_data:
                return {'success': False, 'error': token_data.get('error_description', 'OAuth exchange failed')}
            
            # Exchange Google token for Firebase auth token
            id_token = token_data.get('id_token')
            
            if not id_token:
                return {'success': False, 'error': 'No ID token received from Google'}
            
            # Use Google ID token to sign in with Firebase
            url = f"{self.auth_url}:signInWithIdp?key={self.api_key}"
            firebase_payload = {
                "postBody": f"id_token={id_token}&providerId=google.com",
                "requestUri": redirect_uri,
                "returnIdpCredential": True,
                "returnSecureToken": True
            }
            
            firebase_response = requests.post(url, json=firebase_payload)
            firebase_data = firebase_response.json()
            
            if 'error' in firebase_data:
                return {'success': False, 'error': firebase_data.get('error', {}).get('message', 'Firebase sign-in failed')}
            
            # Successfully authenticated
            return {
                'success': True,
                'user_id': firebase_data.get('localId'),
                'email': firebase_data.get('email'),
                'display_name': firebase_data.get('displayName', ''),
                'token': firebase_data.get('idToken'),
                'refresh_token': firebase_data.get('refreshToken')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
    if 'token_expiry' not in st.session_state:
        st.session_state.token_expiry = None

def logout():
    """Clear session state and logout user"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.auth_token = None
    st.session_state.refresh_token = None
    st.session_state.token_expiry = None
    st.rerun()

def check_token_expiry(firebase_auth):
    """Check if token needs refreshing"""
    if st.session_state.get('token_expiry') and st.session_state.get('refresh_token'):
        if datetime.now() > st.session_state.token_expiry:
            result = firebase_auth.refresh_token(st.session_state.refresh_token)
            if result['success']:
                st.session_state.auth_token = result['token']
                st.session_state.refresh_token = result['refresh_token']
                st.session_state.token_expiry = datetime.now() + timedelta(hours=1)
            else:
                logout()
