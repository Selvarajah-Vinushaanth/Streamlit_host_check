# Firebase Configuration
# Replace these values with your actual Firebase project configuration

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyD2y1V5Tr0DdiPcc1uMrZ5LrdNlpvRh168",
    "authDomain": "song-writing-assistant-4cd39.firebaseapp.com",
    "projectId": "song-writing-assistant-4cd39",
    "storageBucket": "song-writing-assistant-4cd39.firebasestorage.app",
    "messagingSenderId": "74799529758",
    "appId": "1:74799529758:web:472dd9b8b7b1c8af60930f",
    "measurementId": "G-ZPT2GX1FB3",
    # Add Google OAuth credentials
    "googleClientId": "626111935956-aqmgcj4npqtr1d6fr8vcn1e4ifol9h66.apps.googleusercontent.com",
    "googleClientSecret": "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"
}

# Additional configuration
AUTH_CONFIG = {
    "require_email_verification": False,
    "password_min_length": 6,
    "allowed_domains": [],  # Empty list means all domains allowed
    "session_timeout_hours": 24
}

# To get your Firebase configuration:
# 1. Go to Firebase Console (https://console.firebase.google.com/)
# 2. Select your project or create a new one
# 3. Go to Project Settings > General
# 4. Scroll down to "Your apps" section
# 5. Click on "Config" radio button
# 6. Copy the configuration values above

# For service account key:
# 1. Go to Project Settings > Service Accounts
# 2. Click "Generate new private key"
# 3. Download the JSON file and update the path in firebase_auth.py
