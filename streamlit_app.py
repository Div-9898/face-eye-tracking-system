"""
Face & Eye Tracking System - Streamlit App
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Startup check
st.write("🚀 Starting Face & Eye Tracking System...")
st.write(f"Python version: {sys.version}")

try:
    # Import the main app with error handling
    from Video_monitoring_app import StreamlitApp
    
    # Run the application
    if __name__ == "__main__":
        app = StreamlitApp()
        app.run()
        
except ImportError as e:
    st.error("❌ Import Error!")
    st.error(f"Failed to import required modules: {str(e)}")
    st.error("This is likely due to Python version incompatibility.")
    st.info("📝 Solution: Deploy with Python 3.11 in Streamlit Cloud Advanced Settings")
    
except Exception as e:
    st.error("❌ Startup Error!")
    st.error(f"An error occurred during startup: {str(e)}")
    import traceback
    st.code(traceback.format_exc()) 