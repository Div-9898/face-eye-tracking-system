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

# Set page config FIRST before any other st commands
st.set_page_config(
    page_title="Face & Eye Tracking System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Div-9898/face-eye-tracking-system',
        'Report a bug': 'https://github.com/Div-9898/face-eye-tracking-system/issues',
        'About': '# Face & Eye Tracking System\n\nAdvanced real-time facial monitoring with eye tracking and analytics.'
    }
)

# Check Python version first (but display later if there's an issue)
python_version = sys.version_info
has_version_issue = python_version.major == 3 and python_version.minor >= 13

# Try to import dependencies first
import_errors = []
try:
    import cv2
    cv2_version = cv2.__version__
except ImportError as e:
    import_errors.append(("OpenCV", str(e)))
    cv2_version = None

try:
    import mediapipe as mp
    mp_version = mp.__version__
except ImportError as e:
    import_errors.append(("MediaPipe", str(e)))
    mp_version = None

# Now check if we should show startup info or go directly to the app
if has_version_issue or import_errors:
    # Show startup/error information
    st.write("🚀 Starting Face & Eye Tracking System...")
    st.write(f"Python version: {sys.version}")
    
    if has_version_issue:
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
    
    # Show import status
    if cv2_version:
        st.success(f"✓ OpenCV imported successfully (version: {cv2_version})")
    else:
        st.error(f"✗ Failed to import OpenCV")
    
    if mp_version:
        st.success(f"✓ MediaPipe imported successfully (version: {mp_version})")
    else:
        st.error(f"✗ Failed to import MediaPipe")
        if not has_version_issue:
            st.info("This might be due to installation issues. Check the logs above.")
    
    # Show error details
    for package, error in import_errors:
        with st.expander(f"🔍 {package} Error Details"):
            st.code(error)
    
    # Show the current environment details
    with st.expander("🔍 Environment Details"):
        st.code(f"""
Python Version: {sys.version}
Python Executable: {sys.executable}
Platform: {sys.platform}
        """)
    
    st.stop()

# If we get here, everything imported successfully, so try to run the main app
try:
    # Import the main app with error handling
    from Video_monitoring_app import StreamlitApp
    
    # Run the application directly without clearing anything
    if __name__ == "__main__":
        app = StreamlitApp()
        app.run()
        
except ImportError as e:
    st.write("🚀 Starting Face & Eye Tracking System...")
    st.write(f"Python version: {sys.version}")
    st.error("❌ Import Error!")
    st.error(f"Failed to import required modules: {str(e)}")
    st.error("This is likely due to missing dependencies or incompatibility issues.")
    
    import traceback
    with st.expander("🔍 Full Error Trace"):
        st.code(traceback.format_exc())
    
except Exception as e:
    st.write("🚀 Starting Face & Eye Tracking System...")
    st.write(f"Python version: {sys.version}")
    st.error("❌ Startup Error!")
    st.error(f"An error occurred during startup: {str(e)}")
    
    import traceback
    with st.expander("🔍 Full Error Trace"):
        st.code(traceback.format_exc()) 