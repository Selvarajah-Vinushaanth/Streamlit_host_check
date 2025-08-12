import streamlit as st

# Configure Streamlit page - MUST be first Streamlit command
st.set_page_config(
    page_title="Tamil Metaphor Classifier",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth_ui import render_authentication_page
from google_auth_handler import handle_google_callback
import v1

def main():
    """Main application entry point"""
    # Check if this is a Google OAuth callback
    query_params = st.query_params
    
    if 'code' in query_params and 'state' in query_params:
        # This is a Google OAuth callback
        handle_google_callback()
        return
    
    # Handle regular authentication
    is_authenticated = render_authentication_page()
    
    if is_authenticated:
        # Load the main application from v1.py
        try:
            # If v1.py has a main function, call it
            if hasattr(v1, 'main'):
                v1.main()
            # If v1.py has a run function, call it
            elif hasattr(v1, 'run'):
                v1.run()
            # If v1.py has an app function, call it
            elif hasattr(v1, 'app'):
                v1.app()
            else:
                # If no specific function, just import and run v1
                # This will execute the code in v1.py
                st.markdown("### 🎉 Welcome to Tamil Metaphor Classifier!")
                st.info("Loading application from v1.py...")
        except Exception as e:
            st.error(f"❌ Error loading application: {str(e)}")
            st.info("Please check your v1.py file for any errors.")

if __name__ == "__main__":
    main()

