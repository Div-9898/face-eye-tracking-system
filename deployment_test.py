"""
Deployment Test Script
Run this to check if your environment is compatible with the Face & Eye Tracking System
"""

import sys
import platform
import subprocess

print("=" * 60)
print("Face & Eye Tracking System - Deployment Test")
print("=" * 60)

# Python version check
print(f"\n📍 Python Version: {sys.version}")
python_version = sys.version_info

if python_version.major == 3 and python_version.minor >= 13:
    print("❌ ERROR: Python 3.13+ detected!")
    print("   MediaPipe does not support Python 3.13 yet.")
    print("\n   SOLUTION for Streamlit Cloud:")
    print("   1. Go to your app settings")
    print("   2. Select Python 3.11 from the dropdown")
    print("   3. Save and redeploy")
elif python_version.major == 3 and python_version.minor == 11:
    print("✅ Python 3.11 detected - Compatible!")
else:
    print(f"⚠️  Python {python_version.major}.{python_version.minor} detected")
    print("   Recommended: Python 3.11")

# Platform info
print(f"\n📍 Platform: {platform.platform()}")
print(f"📍 Machine: {platform.machine()}")
print(f"📍 Processor: {platform.processor()}")

# Try importing packages
print("\n📦 Checking package compatibility...")

packages = [
    ("streamlit", "Streamlit"),
    ("cv2", "OpenCV"),
    ("mediapipe", "MediaPipe"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("plotly", "Plotly")
]

all_good = True
for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {display_name}: {version}")
    except ImportError as e:
        print(f"❌ {display_name}: Failed to import")
        print(f"   Error: {str(e)}")
        all_good = False

if all_good:
    print("\n🎉 All dependencies are properly installed!")
    print("   Your environment is ready to run the Face & Eye Tracking System.")
else:
    print("\n⚠️  Some dependencies are missing or incompatible.")
    print("\n   For Streamlit Cloud deployment:")
    print("   1. Ensure Python 3.11 is selected in settings")
    print("   2. Check that all packages are listed in requirements.txt")
    print("   3. Verify packages.txt includes system dependencies")

print("\n" + "=" * 60) 