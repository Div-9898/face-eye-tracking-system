import tensorflow as tf
import sys
import platform

print("="*60)
print("TENSORFLOW GPU CONFIGURATION CHECK")
print("="*60)

print(f"\nSystem Information:")
print(f"  Python version: {sys.version}")
print(f"  Platform: {platform.platform()}")
print(f"  TensorFlow version: {tf.__version__}")

print(f"\nGPU Detection:")
gpus = tf.config.list_physical_devices('GPU')
print(f"  Number of GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Device details: {details}")
        except:
            pass
else:
    print("  No GPUs detected by TensorFlow")
    
print(f"\nAll devices:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"  {device}")

print(f"\nCUDA Built: {tf.test.is_built_with_cuda()}")
print(f"GPU Available: {tf.test.is_gpu_available()}")

# Check if CUDA is available
try:
    import subprocess
    print("\nChecking NVIDIA GPU...")
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("NVIDIA GPU found! Output from nvidia-smi:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    else:
        print("nvidia-smi command not found. NVIDIA drivers may not be installed.")
except Exception as e:
    print(f"Could not run nvidia-smi: {e}")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

if not gpus:
    print("To enable GPU support for TensorFlow on Windows:")
    print("1. Install NVIDIA GPU drivers (if you have an NVIDIA GPU)")
    print("2. Install CUDA Toolkit (check TensorFlow GPU requirements)")
    print("3. Install cuDNN (matching your CUDA version)")
    print("4. Reinstall TensorFlow with GPU support: pip install tensorflow-gpu")
    print("\nFor TensorFlow 2.10+ on Windows, you need:")
    print("- CUDA 11.2")
    print("- cuDNN 8.1")
    print("\nVisit: https://www.tensorflow.org/install/gpu")
else:
    print("✓ GPU is available and configured correctly!") 