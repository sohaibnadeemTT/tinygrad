#!/usr/bin/env python3
"""
100M Element Benchmark: Addition vs Pointwise Multiply
Test TTNN vs NumPy performance at massive scale (100M elements)
"""

import os
import time
import numpy as np
from tinygrad import Tensor, Device

def benchmark_operation(operation_name, size=50000000):
    """Benchmark both NumPy and TTNN for a specific operation"""
    
    print(f"\n🚀 BENCHMARKING {operation_name.upper()} - {size:,} ELEMENTS")
    print("=" * 80)
    
    # Test data for the operation
    if operation_name == "addition":
        a_data = np.arange(size, dtype=np.float32) * 100.0
        b_data = np.ones(size, dtype=np.float32) * 200.0
        expected_result = a_data + 200.0
    else:  # multiplication
        a_data = np.arange(size, dtype=np.float32) * 0.1 + 1.0
        b_data = np.ones(size, dtype=np.float32) * 2.5
        expected_result = a_data * 2.5
    
    results = {}
    
    # ========== NUMPY TEST ==========
    print(f"\n🔍 NUMPY (CPU) - {operation_name}")
    print("=" * 50)
    
    print("   📊 Creating numpy arrays...")
    numpy_start = time.time()
    
    if operation_name == "addition":
        numpy_result = a_data + b_data
    else:  # multiplication
        numpy_result = a_data * b_data
        
    numpy_end = time.time()
    numpy_time = (numpy_end - numpy_start) * 1000  # ms
    
    # Accuracy check
    errors = np.abs(numpy_result - expected_result)
    max_error = np.max(errors)
    accurate_count = np.sum(errors < 0.01)
    accuracy = (accurate_count / size) * 100
    throughput = size / numpy_time  # elements/ms
    throughput_millions = throughput / 1000  # millions/second
    
    print(f"   ⚡ Computation time: {numpy_time:.2f}ms")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
    print(f"   📊 Accuracy: {accuracy:.2f}%")
    print(f"   📊 Max error: {max_error:.6f}")
    
    results['numpy'] = {
        'time_ms': numpy_time,
        'throughput_millions': throughput_millions,
        'accuracy': accuracy,
        'max_error': max_error
    }
    
    # ========== TTNN TEST ==========
    print(f"\n🔍 TTNN BACKEND - {operation_name}")
    print("=" * 50)
    
    # Force TTNN
    os.environ["TTNN"] = "1"
    Device.DEFAULT = "TTNN"
    
    print("   📊 Creating test tensors...")
    tensor_start = time.time()
    
    # Convert to Tensors (using numpy arrays directly)
    a = Tensor(a_data)
    b = Tensor(b_data)
    
    print(f"   🔄 Tensors created on device: {a.device}")
    print("   ⏱️  Starting computation...")
    compute_start = time.time()
    
    # Perform operation
    if operation_name == "addition":
        c = a + b
    else:  # multiplication
        c = a * b
    
    # Force computation
    ttnn_result = c.realize().numpy()
    
    compute_end = time.time()
    total_time = (compute_end - tensor_start) * 1000  # ms
    compute_time = (compute_end - compute_start) * 1000  # ms
    
    # Accuracy check
    errors = np.abs(ttnn_result - expected_result)
    max_error = np.max(errors)
    accurate_count = np.sum(errors < 0.01)
    accuracy = (accurate_count / size) * 100
    throughput = size / compute_time  # elements/ms
    throughput_millions = throughput / 1000  # millions/second
    
    print(f"   ⚡ Total time (incl. setup): {total_time:.2f}ms")
    print(f"   ⚡ Pure computation time: {compute_time:.2f}ms")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
    print(f"   📊 Accuracy: {accuracy:.2f}%")
    print(f"   📊 Max error: {max_error:.6f}")
    
    results['ttnn'] = {
        'total_time_ms': total_time,
        'compute_time_ms': compute_time,
        'throughput_millions': throughput_millions,
        'accuracy': accuracy,
        'max_error': max_error
    }
    
    return results

def main():
    print("🚀 50M ELEMENT BENCHMARK: ADDITION vs POINTWISE MULTIPLY")
    print("🎯 Testing TTNN vs NumPy at massive scale")
    print("=" * 80)
    
    size = 50000000  # 50M elements
    
    # Test Addition
    add_results = benchmark_operation("addition", size)
    
    # Test Multiplication  
    multiply_results = benchmark_operation("multiplication", size)
    
    # ========== COMPARISON ==========
    print(f"\n🏆 RESULTS COMPARISON - {size:,} ELEMENTS")
    print("=" * 80)
    
    print("📊 ADDITION RESULTS:")
    print(f"   NumPy:  {add_results['numpy']['throughput_millions']:.2f} M elements/sec ({add_results['numpy']['time_ms']:.2f}ms)")
    print(f"   TTNN:   {add_results['ttnn']['throughput_millions']:.2f} M elements/sec ({add_results['ttnn']['compute_time_ms']:.2f}ms)")
    
    add_speedup = add_results['numpy']['time_ms'] / add_results['ttnn']['compute_time_ms']
    if add_speedup > 1:
        print(f"   🚀 TTNN is {add_speedup:.2f}x FASTER for addition!")
    else:
        print(f"   📊 NumPy is {1/add_speedup:.2f}x faster for addition")
    
    print("\n📊 MULTIPLICATION RESULTS:")
    print(f"   NumPy:  {multiply_results['numpy']['throughput_millions']:.2f} M elements/sec ({multiply_results['numpy']['time_ms']:.2f}ms)")
    print(f"   TTNN:   {multiply_results['ttnn']['throughput_millions']:.2f} M elements/sec ({multiply_results['ttnn']['compute_time_ms']:.2f}ms)")
    
    mult_speedup = multiply_results['numpy']['time_ms'] / multiply_results['ttnn']['compute_time_ms']
    if mult_speedup > 1:
        print(f"   🚀 TTNN is {mult_speedup:.2f}x FASTER for multiplication!")
    else:
        print(f"   📊 NumPy is {1/mult_speedup:.2f}x faster for multiplication")
    
    print("\n🔍 ADD vs MULTIPLY COMPARISON:")
    numpy_add_vs_mult = add_results['numpy']['throughput_millions'] / multiply_results['numpy']['throughput_millions']
    ttnn_add_vs_mult = add_results['ttnn']['throughput_millions'] / multiply_results['ttnn']['throughput_millions']
    
    print(f"   NumPy: Addition is {numpy_add_vs_mult:.2f}x {'faster' if numpy_add_vs_mult > 1 else 'slower'} than multiplication")
    print(f"   TTNN:  Addition is {ttnn_add_vs_mult:.2f}x {'faster' if ttnn_add_vs_mult > 1 else 'slower'} than multiplication")
    
    print(f"\n💡 ANALYSIS AT {size:,} ELEMENTS:")
    if add_speedup > 1 or mult_speedup > 1:
        print("   🎉 TTNN finally shows advantages at massive scale!")
        print("   🔥 Specialized AI hardware dominates for large tensors")
    else:
        print("   📊 NumPy still dominates, even at 50M elements")
        print("   💡 TTNN overhead still significant for simple arithmetic")
    
    print(f"\n✨ 50M ELEMENT BENCHMARK COMPLETE! ✨")

if __name__ == "__main__":
    main()
