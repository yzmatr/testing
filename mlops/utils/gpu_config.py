# utils/gpu_config.py

import tensorflow as tf
        
def setup_gpu(enable_memory_growth=True, log_device_placement=False):
    """
    Configure TensorFlow to optimally use GPU on Apple Silicon (M-chip) MacBooks.
    
    Args:
        enable_memory_growth (bool): Enable memory growth to avoid allocating all GPU memory at once
        log_device_placement (bool): Log which device each operation is placed on
    """
    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"✅ GPU devices available: {[device.name for device in physical_devices]}")
    
    if gpu_devices:
        print(f"📋 Number of GPU devices found: {len(gpu_devices)}")
        for gpu in gpu_devices:
            print(f"  - {gpu.name} ({gpu.device_type})")
            
            # Enable memory growth to avoid allocating all GPU memory at startup
            if enable_memory_growth:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Enabled memory growth for {gpu.name}")
                except RuntimeError as e:
                    print(f"⚠️ Could not set memory growth: {e}")
                    
    else:
        print("⚠️ No GPU devices found. Training will use CPU.")
        
    # Set device placement logging
    if log_device_placement:
        tf.debugging.set_log_device_placement(True)
        print("✅ Enabled device placement logging")
        
    # For Apple Silicon, ensure we're using the Metal backend
    if len(gpu_devices) > 0:
        # Verify we can create a simple operation on GPU
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.linalg.matmul(test_tensor, test_tensor)
                print(f"✅ GPU tested on {result.device} - Metal backend is working.")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            
    #---------------------------------------------------------------------------------------
    # DISPLAY CURRENT CONFIGURATION
    #---------------------------------------------------------------------------------------
    # List all visible devices
    visible_devices = tf.config.get_visible_devices()
    print(f"📝 TensorFlow version: {tf.__version__}")
    print(f"📋 Number of Visible devices: {len(visible_devices)}")
    for device in visible_devices:
        print(f"  - {device.name} ({device.device_type})")
    # Check memory growth setting for GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    for gpu in gpu_devices:
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
            print(f"✅ Memory growth for {gpu.name}: {memory_growth}")
        except Exception as e:
            print(f"❌ Could not check memory growth for {gpu.name}: {e}")
    
def force_cpu():
    """Force TensorFlow to use CPU only (useful for debugging)"""
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("✅ Forced CPU-only mode")
    except Exception as e:
        print(f"❌ Could not set CPU-only mode: {e}")
        
def get_available_gpus():
    """Return list of available GPU devices"""
    return tf.config.list_physical_devices('GPU')

def is_gpu_available():
    """Check if GPU is available and working"""
    gpu_devices = get_available_gpus()
    if not gpu_devices:
        return False
        
    # Test if we can actually use the GPU
    try:
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0])
            _ = tf.reduce_sum(test_tensor)
        return True
    except:
        return False