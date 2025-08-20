#!/usr/bin/env python3
"""
Pure Computation Benchmark: Addition vs Pointwise Multiply
Excludes device initialization and tensor copying overhead
Times only the actual computation operations
"""

import os
import time
import numpy as np
from tinygrad import Tensor, Device

def benchmark_pure_computation(operation_name, size=1000000, num_iterations=10):
    """Benchmark pure computation excluding setup overhead"""
    
    print(f"\n🚀 PURE COMPUTATION BENCHMARK: {operation_name.upper()}")
    print(f"📊 {size:,} elements, {num_iterations} iterations")
    print("=" * 70)
    
    # Test data for the operation
    if operation_name == "addition":
        a_data = np.arange(size, dtype=np.float32) * 100.0
        b_data = np.ones(size, dtype=np.float32) * 200.0
    else:  # multiplication
        a_data = np.arange(size, dtype=np.float32) * 0.1 + 1.0
        b_data = np.ones(size, dtype=np.float32) * 2.5
    
    results = {}
    
    # ========== NUMPY PURE COMPUTATION ==========
    print(f"\n🔍 NUMPY - Pure {operation_name} computation")
    print("-" * 40)
    
    # Single operation timing (most accurate)
    numpy_times = []
    for i in range(num_iterations):
        start_time = time.time()
        
        if operation_name == "addition":
            result = a_data + b_data
        else:  # multiplication
            result = a_data * b_data
            
        end_time = time.time()
        numpy_times.append((end_time - start_time) * 1000)  # ms
    
    avg_numpy_time = np.mean(numpy_times)
    min_numpy_time = np.min(numpy_times)
    throughput = size / avg_numpy_time  # elements/ms
    throughput_millions = throughput / 1000  # millions/second
    
    print(f"   ⚡ Average time: {avg_numpy_time:.3f}ms")
    print(f"   ⚡ Best time: {min_numpy_time:.3f}ms")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
    
    results['numpy'] = {
        'avg_time_ms': avg_numpy_time,
        'min_time_ms': min_numpy_time,
        'throughput_millions': throughput_millions
    }
    
    # ========== TTNN PURE COMPUTATION ==========
    print(f"\n🔍 TTNN - Pure {operation_name} computation")
    print("-" * 40)
    
    # Force TTNN and initialize device ONCE
    os.environ["TTNN"] = "1"
    Device.DEFAULT = "TTNN"
    
    print("   🔧 One-time device initialization...")
    init_start = time.time()
    
    # Create tensors ONCE (exclude from timing)
    a = Tensor(a_data)
    b = Tensor(b_data)
    
    # Force device initialization with a dummy operation
    dummy = a + b
    dummy.realize()
    
    init_time = (time.time() - init_start) * 1000
    print(f"   ✅ Device initialized in {init_time:.2f}ms")
    print(f"   📍 Tensors on device: {a.device}")
    
    # Now time ONLY the pure computation
    print("   ⏱️  Timing pure computation...")
    
    ttnn_times = []
    for i in range(num_iterations):
        # Time ONLY the operation
        start_time = time.time()
        
        if operation_name == "addition":
            result = a + b
        else:  # multiplication
            result = a * b
        
        # Force computation but don't include copying back to CPU
        result.realize()
        
        end_time = time.time()
        ttnn_times.append((end_time - start_time) * 1000)  # ms
    
    avg_ttnn_time = np.mean(ttnn_times)
    min_ttnn_time = np.min(ttnn_times)
    throughput = size / avg_ttnn_time  # elements/ms
    throughput_millions = throughput / 1000  # millions/second
    
    print(f"   ⚡ Average time: {avg_ttnn_time:.3f}ms")
    print(f"   ⚡ Best time: {min_ttnn_time:.3f}ms")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
    
    results['ttnn'] = {
        'avg_time_ms': avg_ttnn_time,
        'min_time_ms': min_ttnn_time,
        'throughput_millions': throughput_millions,
        'init_time_ms': init_time
    }
    
    return results

def main():
    print("🚀 PURE COMPUTATION BENCHMARK: ADDITION vs POINTWISE MULTIPLY")
    print("🎯 Excludes device initialization and tensor copying overhead")
    print("=" * 80)
    
    size = 1000000  # 1M elements
    iterations = 10
    
    # Test Addition
    add_results = benchmark_pure_computation("addition", size, iterations)
    
    # Test Multiplication  
    multiply_results = benchmark_pure_computation("multiplication", size, iterations)
    
    # ========== PURE COMPUTATION COMPARISON ==========
    print(f"\n🏆 PURE COMPUTATION COMPARISON - {size:,} ELEMENTS")
    print("=" * 80)
    
    print("📊 ADDITION (Pure Computation):")
    print(f"   NumPy:  {add_results['numpy']['throughput_millions']:.2f} M elements/sec ({add_results['numpy']['avg_time_ms']:.3f}ms avg)")
    print(f"   TTNN:   {add_results['ttnn']['throughput_millions']:.2f} M elements/sec ({add_results['ttnn']['avg_time_ms']:.3f}ms avg)")
    
    add_speedup = add_results['numpy']['avg_time_ms'] / add_results['ttnn']['avg_time_ms']
    if add_speedup > 1:
        print(f"   🚀 TTNN is {add_speedup:.2f}x FASTER for pure addition!")
    else:
        print(f"   📊 NumPy is {1/add_speedup:.2f}x faster for pure addition")
    
    print("\n📊 MULTIPLICATION (Pure Computation):")
    print(f"   NumPy:  {multiply_results['numpy']['throughput_millions']:.2f} M elements/sec ({multiply_results['numpy']['avg_time_ms']:.3f}ms avg)")
    print(f"   TTNN:   {multiply_results['ttnn']['throughput_millions']:.2f} M elements/sec ({multiply_results['ttnn']['avg_time_ms']:.3f}ms avg)")
    
    mult_speedup = multiply_results['numpy']['avg_time_ms'] / multiply_results['ttnn']['avg_time_ms']
    if mult_speedup > 1:
        print(f"   🚀 TTNN is {mult_speedup:.2f}x FASTER for pure multiplication!")
    else:
        print(f"   📊 NumPy is {1/mult_speedup:.2f}x faster for pure multiplication")
    
    print("\n🔍 OPERATION COMPARISON (Within Each Backend):")
    numpy_add_vs_mult = add_results['numpy']['throughput_millions'] / multiply_results['numpy']['throughput_millions']
    ttnn_add_vs_mult = add_results['ttnn']['throughput_millions'] / multiply_results['ttnn']['throughput_millions']
    
    print(f"   NumPy: Addition is {numpy_add_vs_mult:.2f}x {'faster' if numpy_add_vs_mult > 1 else 'slower'} than multiplication")
    print(f"   TTNN:  Addition is {ttnn_add_vs_mult:.2f}x {'faster' if ttnn_add_vs_mult > 1 else 'slower'} than multiplication")
    
    print(f"\n⚡ BEST CASE PERFORMANCE (min times):")
    add_best_speedup = add_results['numpy']['min_time_ms'] / add_results['ttnn']['min_time_ms']
    mult_best_speedup = multiply_results['numpy']['min_time_ms'] / multiply_results['ttnn']['min_time_ms']
    
    print(f"   Addition: {'TTNN' if add_best_speedup > 1 else 'NumPy'} wins by {max(add_best_speedup, 1/add_best_speedup):.2f}x")
    print(f"   Multiplication: {'TTNN' if mult_best_speedup > 1 else 'NumPy'} wins by {max(mult_best_speedup, 1/mult_best_speedup):.2f}x")
    
    print(f"\n💡 ANALYSIS:")
    if add_speedup > 1 or mult_speedup > 1:
        print("   🎉 TTNN shows computational advantages when overhead is excluded!")
        print("   🔥 Specialized AI hardware dominates for pure computation")
        print(f"   ⚠️  Note: Real-world usage includes {add_results['ttnn']['init_time_ms']:.0f}ms initialization overhead")
    else:
        print("   📊 NumPy still dominates even for pure computation")
        print("   💡 CPU SIMD optimizations very effective for these operations")
        print("   🤔 TTNN may need larger tensors or more complex operations to shine")
    
    print(f"\n✨ PURE COMPUTATION BENCHMARK COMPLETE! ✨")

if __name__ == "__main__":
    main()
