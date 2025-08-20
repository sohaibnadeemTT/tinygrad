#!/usr/bin/env python3
"""
Intensive Benchmark: 100 iterations with 10M elements
Addition vs Pointwise Multiply - Pure computation timing
"""

import os
import time
import numpy as np
from tinygrad import Tensor, Device

def benchmark_intensive(operation_name, size=10000000, num_iterations=100):
    """Intensive benchmark with large tensor and many iterations"""
    
    print(f"\n🚀 INTENSIVE BENCHMARK: {operation_name.upper()}")
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
    
    # ========== NUMPY INTENSIVE TEST ==========
    print(f"\n🔍 NUMPY - {num_iterations} iterations of {operation_name}")
    print("-" * 50)
    
    numpy_times = []
    print("   ⏱️  Running iterations...", end="", flush=True)
    
    for i in range(num_iterations):
        if i % 10 == 0:
            print(f"{i}...", end="", flush=True)
            
        start_time = time.time()
        
        if operation_name == "addition":
            result = a_data + b_data
        else:  # multiplication
            result = a_data * b_data
            
        end_time = time.time()
        numpy_times.append((end_time - start_time) * 1000)  # ms
    
    print("✅")
    
    avg_numpy_time = np.mean(numpy_times)
    min_numpy_time = np.min(numpy_times)
    max_numpy_time = np.max(numpy_times)
    std_numpy_time = np.std(numpy_times)
    throughput = size / avg_numpy_time  # elements/ms
    throughput_millions = throughput / 1000  # millions/second
    
    print(f"   ⚡ Average time: {avg_numpy_time:.3f}ms (±{std_numpy_time:.3f}ms)")
    print(f"   ⚡ Best time: {min_numpy_time:.3f}ms")
    print(f"   ⚡ Worst time: {max_numpy_time:.3f}ms")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
    
    results['numpy'] = {
        'avg_time_ms': avg_numpy_time,
        'min_time_ms': min_numpy_time,
        'max_time_ms': max_numpy_time,
        'std_time_ms': std_numpy_time,
        'throughput_millions': throughput_millions,
        'all_times': numpy_times
    }
    
    # ========== TTNN INTENSIVE TEST ==========
    print(f"\n🔍 TTNN - {num_iterations} iterations of {operation_name}")
    print("-" * 50)
    
    # Force TTNN and initialize device ONCE
    os.environ["TTNN"] = "1"
    Device.DEFAULT = "TTNN"
    
    print("   🔧 One-time device initialization...")
    init_start = time.time()
    
    try:
        # Create tensors ONCE (exclude from timing)
        a = Tensor(a_data)
        b = Tensor(b_data)
        
        # Force device initialization with a dummy operation
        dummy = a + b
        dummy.realize()
        
        init_time = (time.time() - init_start) * 1000
        print(f"   ✅ Device initialized in {init_time:.2f}ms")
        print(f"   📍 Tensors on device: {a.device}")
        
        # Now time the intensive computation
        print("   ⏱️  Running intensive computation...", end="", flush=True)
        
        ttnn_times = []
        for i in range(num_iterations):
            if i % 10 == 0:
                print(f"{i}...", end="", flush=True)
                
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
        
        print("✅")
        
        avg_ttnn_time = np.mean(ttnn_times)
        min_ttnn_time = np.min(ttnn_times)
        max_ttnn_time = np.max(ttnn_times)
        std_ttnn_time = np.std(ttnn_times)
        throughput = size / avg_ttnn_time  # elements/ms
        throughput_millions = throughput / 1000  # millions/second
        
        print(f"   ⚡ Average time: {avg_ttnn_time:.3f}ms (±{std_ttnn_time:.3f}ms)")
        print(f"   ⚡ Best time: {min_ttnn_time:.3f}ms")
        print(f"   ⚡ Worst time: {max_ttnn_time:.3f}ms")
        print(f"   ⚡ Throughput: {throughput_millions:.2f} M elements/sec")
        
        results['ttnn'] = {
            'avg_time_ms': avg_ttnn_time,
            'min_time_ms': min_ttnn_time,
            'max_time_ms': max_ttnn_time,
            'std_time_ms': std_ttnn_time,
            'throughput_millions': throughput_millions,
            'init_time_ms': init_time,
            'all_times': ttnn_times,
            'success': True
        }
        
    except Exception as e:
        print(f"\n   💥 TTNN failed: {str(e)}")
        if "Out of Memory" in str(e):
            print("   📊 Memory limit exceeded - tensor too large for TTNN device")
        results['ttnn'] = {'success': False, 'error': str(e)}
    
    return results

def main():
    print("🚀 INTENSIVE BENCHMARK: 100 ITERATIONS × 10M ELEMENTS")
    print("🎯 Testing sustained performance under load")
    print("=" * 80)
    
    size = 10000000  # 10M elements
    iterations = 100
    
    # Test Addition
    print("\n🧪 TESTING ADDITION")
    add_results = benchmark_intensive("addition", size, iterations)
    
    # Test Multiplication  
    print("\n🧪 TESTING MULTIPLICATION")
    multiply_results = benchmark_intensive("multiplication", size, iterations)
    
    # ========== INTENSIVE BENCHMARK RESULTS ==========
    print(f"\n🏆 INTENSIVE BENCHMARK RESULTS - {size:,} ELEMENTS × {iterations} ITERATIONS")
    print("=" * 80)
    
    print("📊 ADDITION PERFORMANCE:")
    print(f"   NumPy:  {add_results['numpy']['throughput_millions']:.2f} M elements/sec")
    print(f"          avg: {add_results['numpy']['avg_time_ms']:.3f}ms ± {add_results['numpy']['std_time_ms']:.3f}ms")
    print(f"          range: {add_results['numpy']['min_time_ms']:.3f}ms - {add_results['numpy']['max_time_ms']:.3f}ms")
    
    if add_results['ttnn']['success']:
        print(f"   TTNN:   {add_results['ttnn']['throughput_millions']:.2f} M elements/sec")
        print(f"          avg: {add_results['ttnn']['avg_time_ms']:.3f}ms ± {add_results['ttnn']['std_time_ms']:.3f}ms")
        print(f"          range: {add_results['ttnn']['min_time_ms']:.3f}ms - {add_results['ttnn']['max_time_ms']:.3f}ms")
        
        add_speedup = add_results['numpy']['avg_time_ms'] / add_results['ttnn']['avg_time_ms']
        if add_speedup > 1:
            print(f"   🚀 TTNN is {add_speedup:.2f}x FASTER for addition!")
        else:
            print(f"   📊 NumPy is {1/add_speedup:.2f}x faster for addition")
    else:
        print(f"   TTNN:   FAILED - {add_results['ttnn']['error']}")
    
    print("\n📊 MULTIPLICATION PERFORMANCE:")
    print(f"   NumPy:  {multiply_results['numpy']['throughput_millions']:.2f} M elements/sec")
    print(f"          avg: {multiply_results['numpy']['avg_time_ms']:.3f}ms ± {multiply_results['numpy']['std_time_ms']:.3f}ms")
    print(f"          range: {multiply_results['numpy']['min_time_ms']:.3f}ms - {multiply_results['numpy']['max_time_ms']:.3f}ms")
    
    if multiply_results['ttnn']['success']:
        print(f"   TTNN:   {multiply_results['ttnn']['throughput_millions']:.2f} M elements/sec")
        print(f"          avg: {multiply_results['ttnn']['avg_time_ms']:.3f}ms ± {multiply_results['ttnn']['std_time_ms']:.3f}ms")
        print(f"          range: {multiply_results['ttnn']['min_time_ms']:.3f}ms - {multiply_results['ttnn']['max_time_ms']:.3f}ms")
        
        mult_speedup = multiply_results['numpy']['avg_time_ms'] / multiply_results['ttnn']['avg_time_ms']
        if mult_speedup > 1:
            print(f"   🚀 TTNN is {mult_speedup:.2f}x FASTER for multiplication!")
        else:
            print(f"   📊 NumPy is {1/mult_speedup:.2f}x faster for multiplication")
    else:
        print(f"   TTNN:   FAILED - {multiply_results['ttnn']['error']}")
    
    # Performance stability analysis
    if add_results['ttnn']['success'] and multiply_results['ttnn']['success']:
        print("\n📈 PERFORMANCE STABILITY ANALYSIS:")
        
        numpy_add_cv = add_results['numpy']['std_time_ms'] / add_results['numpy']['avg_time_ms'] * 100
        ttnn_add_cv = add_results['ttnn']['std_time_ms'] / add_results['ttnn']['avg_time_ms'] * 100
        
        numpy_mult_cv = multiply_results['numpy']['std_time_ms'] / multiply_results['numpy']['avg_time_ms'] * 100
        ttnn_mult_cv = multiply_results['ttnn']['std_time_ms'] / multiply_results['ttnn']['avg_time_ms'] * 100
        
        print(f"   NumPy Consistency:")
        print(f"     Addition: {numpy_add_cv:.1f}% variation (CV)")
        print(f"     Multiplication: {numpy_mult_cv:.1f}% variation (CV)")
        
        print(f"   TTNN Consistency:")
        print(f"     Addition: {ttnn_add_cv:.1f}% variation (CV)")
        print(f"     Multiplication: {ttnn_mult_cv:.1f}% variation (CV)")
        
        print("\n🔍 OPERATION COMPARISON (Within Each Backend):")
        numpy_add_vs_mult = add_results['numpy']['throughput_millions'] / multiply_results['numpy']['throughput_millions']
        ttnn_add_vs_mult = add_results['ttnn']['throughput_millions'] / multiply_results['ttnn']['throughput_millions']
        
        print(f"   NumPy: Addition is {numpy_add_vs_mult:.2f}x {'faster' if numpy_add_vs_mult > 1 else 'slower'} than multiplication")
        print(f"   TTNN:  Addition is {ttnn_add_vs_mult:.2f}x {'faster' if ttnn_add_vs_mult > 1 else 'slower'} than multiplication")
    
    print(f"\n✨ INTENSIVE BENCHMARK COMPLETE! ✨")
    print(f"📊 Total operations performed: {(iterations * 2):,} ({iterations:,} per operation type)")

if __name__ == "__main__":
    main()
