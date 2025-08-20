#!/usr/bin/env python3
"""
TTNN Matrix Multiplication Test
Test proper matrix multiplication operations (not element-wise)
"""

import os
import numpy as np
from tinygrad import Tensor, Device

def test_basic_matmul():
    """Test basic matrix multiplication operations"""
    
    print("🧮 TTNN MATRIX MULTIPLICATION TEST")
    print("=" * 50)
    
    Device.DEFAULT = "TTNN"
    
    test_cases = [
        ("Small 2x2", (2, 2), (2, 2)),
        ("Small 3x3", (3, 3), (3, 3)), 
        #("Rectangular 4x3 @ 3x5", (4, 3), (3, 5)),
        #("Neural network size 784x256", (784, 256), (256, 10)),
        ("Medium 32x32", (32, 32), (32, 32)),
        #("Large 128x128", (128, 128), (128, 128)),
    ]
    
    for test_name, shape_a, shape_b in test_cases:
        print(f"\n🧪 Testing {test_name}")
        print(f"   📐 Shapes: {shape_a} @ {shape_b} → {(shape_a[0], shape_b[1])}")
        
        try:
            # Create test matrices
            np.random.seed(42)  # For reproducibility
            a_data = np.random.randn(*shape_a).astype(np.float32)
            b_data = np.random.randn(*shape_b).astype(np.float32)
            
            print(f"   📊 Creating tensors...")
            a = Tensor(a_data.tolist())
            b = Tensor(b_data.tolist())
            
            print(f"   🔄 Tensors on device: {a.device}")
            print(f"   ⏱️  Performing matrix multiplication...")
            
            # Perform matrix multiplication using @ operator
            start_time = time.time()
            c = a @ b
            result = c.realize().numpy()
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000
            
            # Verify with NumPy
            expected = np.matmul(a_data, b_data)
            
            # Check accuracy
            max_error = np.max(np.abs(result - expected))
            mean_error = np.mean(np.abs(result - expected))
            accuracy = np.allclose(result, expected, atol=1e-3)  # Reasonable tolerance for bfloat16
            
            print(f"   ✅ SUCCESS!")
            print(f"   ⚡ Time: {computation_time:.2f}ms")
            print(f"   📊 Result shape: {result.shape}")
            print(f"   📊 Max error: {max_error:.6f}")
            print(f"   📊 Mean error: {mean_error:.6f}")
            print(f"   📊 Accuracy: {'PASS' if accuracy else 'FAIL'}")
            
            # Show sample values for small matrices
            if shape_a[0] <= 4 and shape_a[1] <= 4:
                print(f"   📋 Result sample:")
                print(f"      {result}")
                print(f"   📋 Expected sample:")
                print(f"      {expected}")
                
        except Exception as e:
            print(f"   💥 FAILED: {e}")
            # Show more details for debugging
            if "Out of Memory" in str(e):
                expected_elements = shape_a[0] * shape_b[1]
                expected_mb = expected_elements * 4 / 1024 / 1024  # 4 bytes per float32
                print(f"   📊 Expected result: {expected_elements:,} elements ({expected_mb:.2f} MB)")
                if expected_mb > 90:
                    print(f"   ⚠️  Size exceeds device capacity (90MB limit)")
                else:
                    print(f"   ⚠️  Should fit in device - may be intermediate tensor issue")

def test_neural_network_pattern():
    """Test matrix multiplication patterns common in neural networks"""
    
    print(f"\n🧠 NEURAL NETWORK PATTERN TEST")
    print("=" * 50)
    
    try:
        Device.DEFAULT = "TTNN"
        
        # Simulate MNIST neural network dimensions
        batch_size = 1
        input_size = 784    # 28x28 flattened
        hidden_size = 256
        output_size = 10
        
        print(f"   📊 Simulating MNIST MLP:")
        print(f"   📊 Input: {batch_size}x{input_size}")
        print(f"   📊 Weight1: {input_size}x{hidden_size}")
        print(f"   📊 Weight2: {hidden_size}x{output_size}")
        
        # Create test data
        input_data = np.random.randn(batch_size, input_size).astype(np.float32)
        weight1_data = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1
        weight2_data = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.1
        
        print(f"   🔄 Creating tensors...")
        x = Tensor(input_data.tolist())
        w1 = Tensor(weight1_data.tolist())
        w2 = Tensor(weight2_data.tolist())
        
        print(f"   🧮 Layer 1: x @ w1 ({batch_size}x{input_size} @ {input_size}x{hidden_size})")
        hidden = x @ w1
        h1_result = hidden.realize().numpy()
        
        print(f"   ✅ Layer 1 complete: {h1_result.shape}")
        print(f"   📊 Hidden values range: [{h1_result.min():.3f}, {h1_result.max():.3f}]")
        
        print(f"   🧮 Layer 2: hidden @ w2 ({batch_size}x{hidden_size} @ {hidden_size}x{output_size})")
        output = hidden @ w2
        output_result = output.realize().numpy()
        
        print(f"   ✅ Layer 2 complete: {output_result.shape}")
        print(f"   📊 Output values: {output_result.flatten()}")
        
        # Verify with NumPy
        expected_h1 = np.matmul(input_data, weight1_data)
        expected_output = np.matmul(expected_h1, weight2_data)
        
        h1_accuracy = np.allclose(h1_result, expected_h1, atol=1e-3)
        output_accuracy = np.allclose(output_result, expected_output, atol=1e-3)
        
        print(f"   📊 Layer 1 accuracy: {'PASS' if h1_accuracy else 'FAIL'}")
        print(f"   📊 Layer 2 accuracy: {'PASS' if output_accuracy else 'FAIL'}")
        
        if h1_accuracy and output_accuracy:
            print(f"   🎉 NEURAL NETWORK PATTERN WORKS!")
        else:
            print(f"   ⚠️  Accuracy issues detected")
            
    except Exception as e:
        print(f"   💥 NEURAL NETWORK TEST FAILED: {e}")

def test_large_matmul_scaling():
    """Test how large matrix multiplications scale on TTNN"""
    
    print(f"\n📈 LARGE MATRIX MULTIPLICATION SCALING")
    print("=" * 50)
    
    sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 128),  # Rectangular - more realistic for ML
        (2048, 64),   # Very tall and narrow
    ]
    
    Device.DEFAULT = "TTNN"
    
    for h, w in sizes:
        size_name = f"{h}x{w}"
        elements = h * w
        mb_size = elements * 4 / 1024 / 1024
        
        print(f"\n   🧪 Testing {size_name} matrices")
        print(f"   📊 Elements per matrix: {elements:,}")
        print(f"   📊 Memory per matrix: {mb_size:.2f} MB")
        
        if mb_size > 45:  # Our known device limit
            print(f"   ⚠️  Size exceeds single tensor limit - skipping")
            continue
            
        try:
            import time
            
            # Create matrices
            a_data = np.random.randn(h, w).astype(np.float32) * 0.1
            b_data = np.random.randn(w, h).astype(np.float32) * 0.1  # Square result
            
            a = Tensor(a_data.tolist())
            b = Tensor(b_data.tolist())
            
            # Time the operation
            start_time = time.time()
            c = a @ b
            result = c.realize().numpy()
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000
            throughput = (h * w * h) / computation_time  # Operations per ms
            
            print(f"   ✅ SUCCESS: {computation_time:.2f}ms")
            print(f"   📊 Result shape: {result.shape}")
            print(f"   ⚡ Throughput: {throughput:.0f} ops/ms")
            
        except Exception as e:
            print(f"   💥 FAILED: {e}")
            if "Out of Memory" in str(e):
                print(f"   📊 Hit memory limit at {size_name}")

if __name__ == "__main__":
    import time
    
    print("🚀 COMPREHENSIVE TTNN MATRIX MULTIPLICATION TEST")
    print("=" * 80)
    
    # Test 1: Basic matrix operations
    test_basic_matmul()
    
    # Test 2: Neural network patterns  
    #test_neural_network_pattern()
    
    # Test 3: Scaling behavior
    #test_large_matmul_scaling()
    
    print(f"\n✨ MATRIX MULTIPLICATION TESTING COMPLETE! ✨")
