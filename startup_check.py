import sys
import platform

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")

# Check if we can import basic packages
try:
    import streamlit
    print(f"✓ Streamlit version: {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import mediapipe
    print(f"✓ MediaPipe version: {mediapipe.__version__}")
except ImportError as e:
    print(f"✗ MediaPipe import failed: {e}")

print("\nStartup check complete!") 