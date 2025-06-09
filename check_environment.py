"""
Environment Check for Face & Eye Tracking System
This app checks if the Python environment is compatible with MediaPipe
"""

import streamlit as st
import sys
import platform

st.set_page_config(page_title="Environment Check", page_icon="🔍", layout="wide")

st.title("🔍 Environment Check for Face & Eye Tracking System")

# Check Python version
python_version = sys.version_info
is_compatible = python_version.major == 3 and python_version.minor == 11

st.header("Python Version Check")
col1, col2 = st.columns(2)

with col1:
    st.metric("Current Python Version", f"{python_version.major}.{python_version.minor}.{python_version.micro}")
    st.metric("Required Python Version", "3.11.x")

with col2:
    if is_compatible:
        st.success("✅ Python version is compatible!")
    else:
        st.error("❌ Python version is NOT compatible!")

# Detailed information
st.header("System Information")
st.code(f"""
Python Version: {sys.version}
Python Executable: {sys.executable}
Platform: {platform.platform()}
Machine: {platform.machine()}
""")

if not is_compatible:
    st.header("🔧 How to Fix This")
    
    st.error("**MediaPipe does NOT support Python 3.13+**")
    
    st.markdown("""
    ### Option 1: Change Python Version in Current App
    
    1. Go to your [Streamlit Cloud dashboard](https://share.streamlit.io/)
    2. Find your app: **face-eye-tracking-system**
    3. Click the **three dots menu (⋮)**
    4. Select **"Settings"**
    5. Find **"Python version"** setting
    6. Change from **"3.13"** to **"3.11"**
    7. Click **"Save"**
    8. Wait for the app to redeploy
    
    ### Option 2: Delete and Redeploy
    
    1. Go to your [Streamlit Cloud dashboard](https://share.streamlit.io/)
    2. Delete the current app
    3. Click **"New app"**
    4. Select your repository
    5. **IMPORTANT**: Click **"Advanced settings"**
    6. Select **Python 3.11** from the dropdown
    7. Deploy
    
    ### Option 3: Use Alternative Package
    
    If you absolutely must use Python 3.13, you could try:
    - Using a different face tracking library
    - Waiting for MediaPipe to support Python 3.13
    - Using a containerized solution
    """)
    
    # Show a visual guide
    st.info("📸 Here's what the Python version setting looks like in Streamlit Cloud:")
    st.markdown("""
    ```
    Advanced settings
    ├── Python version: [3.11 ▼]  <-- Select this
    │   ├── 3.8
    │   ├── 3.9
    │   ├── 3.10
    │   ├── 3.11  ✓
    │   ├── 3.12
    │   └── 3.13  ✗ (Current - NOT compatible)
    ```
    """)

# Test imports
st.header("📦 Package Import Test")

packages = [
    ("streamlit", "Streamlit"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas")
]

for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        st.success(f"✅ {display_name}: {version}")
    except ImportError as e:
        st.error(f"❌ {display_name}: Failed to import - {str(e)}")

# MediaPipe specific test
st.subheader("MediaPipe Compatibility Test")
try:
    import mediapipe as mp
    st.success(f"✅ MediaPipe: {mp.__version__}")
except ImportError as e:
    st.error(f"❌ MediaPipe: Cannot import - {str(e)}")
    st.warning("This is expected if Python version is not 3.11")

st.divider()

st.markdown("""
### 📝 Summary

Your app is failing because:
1. Streamlit Cloud is using Python 3.13.3
2. MediaPipe only supports up to Python 3.11
3. The `runtime.txt` file is being ignored

**You MUST manually change the Python version in Streamlit Cloud settings to fix this.**
""")

# Add a button to check for updates
if st.button("🔄 Refresh Page"):
    st.rerun() 