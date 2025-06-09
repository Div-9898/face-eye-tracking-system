"""
FIX TENSORFLOW GPU SUPPORT ON WINDOWS

Your system has:
- NVIDIA GPU (detected by nvidia-smi)
- CUDA 12.9 (too new for TensorFlow 2.18)
- TensorFlow 2.18.0 CPU-only version

SOLUTION:
"""

print("="*70)
print("TENSORFLOW GPU INSTALLATION GUIDE FOR WINDOWS")
print("="*70)

print("\nCURRENT ISSUE:")
print("- You have an NVIDIA GPU but TensorFlow is using CPU only")
print("- Your CUDA version (12.9) is too new for TensorFlow 2.18")

print("\nSOLUTION OPTIONS:")
print("\n1. OPTION A: Use TensorFlow with DirectML (Recommended for Windows)")
print("   This works with any DirectX 12 compatible GPU (NVIDIA, AMD, Intel)")
print("   Installation:")
print("   pip uninstall tensorflow")
print("   pip install tensorflow-directml")

print("\n2. OPTION B: Use WSL2 with GPU support")
print("   - Install WSL2 with Ubuntu")
print("   - Install CUDA toolkit in WSL2")
print("   - Use TensorFlow GPU in WSL2 environment")

print("\n3. OPTION C: Downgrade to compatible versions")
print("   For TensorFlow 2.10+ on native Windows, you need:")
print("   - CUDA 11.2")
print("   - cuDNN 8.1")
print("   But this requires uninstalling CUDA 12.9")

print("\n" + "="*70)
print("QUICK FIX - TRY TENSORFLOW-DIRECTML:")
print("="*70)

import subprocess
import sys

response = input("\nWould you like to install tensorflow-directml now? (y/n): ")
if response.lower() == 'y':
    print("\nUninstalling current TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
    
    print("\nInstalling TensorFlow-DirectML...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-directml"])
    
    print("\n✓ Installation complete!")
    print("\nTo verify GPU support, run:")
    print("python -c \"import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))\"")
else:
    print("\nTo install manually, run:")
    print("pip uninstall tensorflow")
    print("pip install tensorflow-directml") 