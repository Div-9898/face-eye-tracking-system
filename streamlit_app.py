"""
Face & Eye Tracking System - Streamlit App
Main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os
import subprocess

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Display startup information
st.set_page_config(page_title="Face & Eye Tracking System", page_icon="👁️", layout="wide")
st.write("🚀 Starting Face & Eye Tracking System...")
st.write(f"Python version: {sys.version}")

# Check Python version
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 13:
    st.error("❌ Python Version Incompatibility Detected!")
    st.error(f"Current Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    st.error("MediaPipe does not support Python 3.13+ yet.")
    
    st.info("### 🔧 How to Fix This Issue:")
    st.markdown("""
    1. **In Streamlit Cloud:**
       - Go to your app dashboard
       - Click on the **⚙️ Settings** button (three dots menu)
       - Select **Settings** → **General**
       - Click on **Python version**
       - Select **Python 3.11** from the dropdown
       - Click **Save**
       - The app will automatically redeploy with Python 3.11
    
    2. **Alternative: Fork and Redeploy**
       - Fork this repository
       - Deploy the forked version
       - During deployment, expand **Advanced settings**
       - Select **Python 3.11** before deploying
    """)
    
    st.warning("📝 Note: runtime.txt and .python-version files have been added to the repository, but Streamlit Cloud requires manual Python version selection in the settings.")
    
    # Show the current environment details
    with st.expander("🔍 Environment Details"):
        st.code(f"""
Python Version: {sys.version}
Python Executable: {sys.executable}
Platform: {sys.platform}
        """)
    
    st.stop()

try:
    # Try to import dependencies
    import cv2
    st.success(f"✓ OpenCV imported successfully (version: {cv2.__version__})")
except ImportError as e:
    st.error(f"✗ Failed to import OpenCV: {str(e)}")

try:
    import mediapipe as mp
    st.success(f"✓ MediaPipe imported successfully (version: {mp.__version__})")
except ImportError as e:
    st.error(f"✗ Failed to import MediaPipe: {str(e)}")
    st.info("This is likely due to Python version incompatibility. Please set Python version to 3.11 in Streamlit Cloud settings.")

try:
    # Import the main app with error handling
    from Video_monitoring_app import StreamlitApp
    
    # Clear the startup messages and run the app
    st.empty()
    
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