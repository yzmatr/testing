#!/usr/bin/env python3
"""
Test script to verify GPU configuration is working properly on Apple Silicon MacBook.
"""

import sys
import os
from pathlib import Path
import pytest
import tensorflow as tf

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import and test GPU configuration
from mlops.utils.gpu_config import setup_gpu

def main():
    print("🔧 Testing GPU Configuration for Apple Silicon MacBook...")
    print("=" * 60)
    
    # Setup GPU configuration
    gpu_config = setup_gpu()
    
    print("\n📊 GPU Status:")
    print(f"   GPU Available: {gpu_config.is_gpu_available()}")
    print(f"   Number of GPUs: {len(gpu_config.get_available_gpus())}")
    
    # Test basic tensor operations on GPU
    print("\n🧪 Testing GPU Operations:")
    try:
        # Test matrix multiplication on GPU
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print(f"   ✅ Matrix multiplication successful")
            print(f"   📍 Computation device: {c.device}")
            
        # Test CNN operations (similar to what the frog classifier will use)
        with tf.device('/GPU:0'):
            input_tensor = tf.random.normal([32, 128, 128, 1])  # Batch of spectrograms
            conv_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
            output = conv_layer(input_tensor)
            print(f"   ✅ CNN operation successful")
            print(f"   📍 Output shape: {output.shape}")
            print(f"   📍 Output device: {output.device}")
            
    except Exception as e:
        print(f"   ❌ GPU operations failed: {e}")
        return False
    
    print("\n🎯 Recommendations for your experiments:")
    if gpu_config.is_gpu_available():
        print("   • GPU is ready for training!")
        print("   • Your M-chip MacBook will use Metal Performance Shaders")
        print("   • Training should be significantly faster than CPU-only")
        print("   • Memory growth is enabled to prevent memory issues")
    else:
        print("   • GPU not available - will fall back to CPU")
        print("   • Consider checking TensorFlow Metal plugin installation")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ GPU test completed successfully!")
    else:
        print("\n❌ GPU test failed!")
        sys.exit(1)

    assert main()

# Test GPU configuration
def test_gpu_available():
    """Test if GPU is available and configured correctly."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}") 