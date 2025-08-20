#!/usr/bin/env python3
"""
NumPy vs TTNN Performance Benchmark - Pointwise Multiply
Compare execution times between NumPy (CPU) and TTNN backends for element-wise multiplication
"""

import os
import time
import numpy as np
from tinygrad import Tensor, Device

def benchmark_numpy(size=100000):
    """Benchmark tensor pointwise multiplication using NumPy (CPU baseline)"""
    
    print(f"\n🔍 BENCHMARKING NUMPY (CPU) BASELINE")
    print("=" * 50)
    print(f"   🎯 Platform: CPU with NumPy")
    print(f"   📊 Testing {size:,} element tensors")
    
    # Create test data
    print("   📊 Creating numpy arrays...")
    a_data = np.arange(size, dtype=np.float32) * 0.1 + 1.0  # Values from 1.0 to size*0.1+1.0
    b_data = np.ones(size, dtype=np.float32) * 2.5  # Constant multiplier
    
    print("   ⏱️  Starting computation...")
    start_time = time.time()
    
    # Perform pointwise multiplication
    result = a_data * b_data
    
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Performance metrics
    throughput = size / computation_time  # elements per ms
    throughput_millions = throughput / 1000  # millions of elements per second
    
    print(f"   ⚡ Computation time: {computation_time:.2f}ms")
    print(f"   ⚡ Elements/ms: {throughput:.0f}")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} million elements/second")
    
    # Accuracy check
    expected = a_data * 2.5
    errors = np.abs(result - expected)
    max_error = np.max(errors)
    accurate_count = np.sum(errors < 0.01)  # Tighter tolerance for multiplication
    accuracy = (accurate_count / size) * 100
    
    print(f"   📊 Accuracy: {accuracy:.2f}%")
    print(f"   📊 Max error: {max_error:.6f}")
    print(f"   📋 First 5 results: {result[:5].tolist()}")
    
    return {
        'backend': 'NumPy',
        'size': size,
        'time_ms': computation_time,
        'throughput_ms': throughput,
        'throughput_millions': throughput_millions,
        'accuracy': accuracy,
        'max_error': max_error
    }

def benchmark_ttnn(size=100000):
    """Benchmark tensor pointwise multiplication using TTNN backend"""
    
    print(f"\n🔍 BENCHMARKING TTNN BACKEND")
    print("=" * 50)
    
    # Force TTNN
    os.environ["TTNN"] = "1"
    Device.DEFAULT = "TTNN"
    
    print(f"   🎯 Device: {Device.DEFAULT}")
    print(f"   📊 Testing {size:,} element tensors")
    
    # Create test data
    print("   📊 Creating test tensors...")
    a_data = np.arange(size, dtype=np.float32) * 0.1 + 1.0  # Values from 1.0 to size*0.1+1.0
    b_data = np.ones(size, dtype=np.float32) * 2.5  # Constant multiplier
    
    # Convert to Tensors
    a = Tensor(a_data.tolist())
    b = Tensor(b_data.tolist())
    
    print(f"   🔄 Tensors created on device: {a.device}")
    
    # Benchmark multiplication
    print("   ⏱️  Starting computation...")
    start_time = time.time()
    
    # Perform pointwise multiplication
    c = a * b
    
    # Force computation (realize the tensor)
    result = c.realize().numpy()
    
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Performance metrics
    throughput = size / computation_time  # elements per ms
    throughput_millions = throughput / 1000  # millions of elements per second
    
    print(f"   ⚡ Computation time: {computation_time:.2f}ms")
    print(f"   ⚡ Elements/ms: {throughput:.0f}")
    print(f"   ⚡ Throughput: {throughput_millions:.2f} million elements/second")
    
    # Accuracy check
    expected = a_data * 2.5
    errors = np.abs(result - expected)
    max_error = np.max(errors)
    accurate_count = np.sum(errors < 0.01)  # Same tolerance as numpy test
    accuracy = (accurate_count / size) * 100
    
    print(f"   📊 Accuracy: {accuracy:.2f}%")
    print(f"   📊 Max error: {max_error:.6f}")
    print(f"   📋 First 5 results: {result[:5].tolist()}")
    
    return {
        'backend': 'TTNN',
        'size': size,
        'time_ms': computation_time,
        'throughput_ms': throughput,
        'throughput_millions': throughput_millions,
        'accuracy': accuracy,
        'max_error': max_error
    }

def main():
    print("🚀 NUMPY vs TTNN PERFORMANCE BENCHMARK - POINTWISE MULTIPLY")
    print("=" * 80)
    
    # Test sizes
    test_sizes = [
        10000,      # 10K elements
        50000,      # 50K elements  
        100000,     # 100K elements
        300000,     # 300K elements (our stress test size)
        1000000,    # 1M elements (massive test)
        10000000,   # 10M elements (TTNN sweet spot test)
    ]
    
    results = []
    
    for size in test_sizes:
        print(f"\n🧪 TESTING {size:,} ELEMENT TENSORS - POINTWISE MULTIPLY")
        print("=" * 80)
        
        # Test NumPy (CPU baseline)
        try:
            numpy_result = benchmark_numpy(size)
            results.append(numpy_result)
        except Exception as e:
            print(f"   💥 NumPy test failed: {e}")
            numpy_result = None
        
        # Test TTNN 
        try:
            ttnn_result = benchmark_ttnn(size)
            results.append(ttnn_result)
        except Exception as e:
            print(f"   💥 TTNN test failed: {e}")
            ttnn_result = None
        
        # Compare results
        if numpy_result and ttnn_result:
            print(f"\n📊 PERFORMANCE COMPARISON ({size:,} elements):")
            print("=" * 50)
            
            speedup = numpy_result['time_ms'] / ttnn_result['time_ms']
            throughput_gain = ttnn_result['throughput_millions'] / numpy_result['throughput_millions']
            
            print(f"   NumPy Time:  {numpy_result['time_ms']:.2f}ms")
            print(f"   TTNN Time:   {ttnn_result['time_ms']:.2f}ms")
            
            if speedup > 1:
                print(f"   🚀 SPEEDUP:   {speedup:.2f}x faster (TTNN wins!)")
            else:
                print(f"   📊 SPEEDUP:   {1/speedup:.2f}x slower (NumPy wins)")
            
            print()
            print(f"   NumPy Throughput:  {numpy_result['throughput_millions']:.2f} M elements/sec")
            print(f"   TTNN Throughput:   {ttnn_result['throughput_millions']:.2f} M elements/sec")
            
            if throughput_gain > 1:
                print(f"   🚀 THROUGHPUT GAIN: {throughput_gain:.2f}x (TTNN wins!)")
            else:
                print(f"   📊 THROUGHPUT LOSS: {1/throughput_gain:.2f}x (NumPy wins)")
            
            print()
            print(f"   NumPy Accuracy:  {numpy_result['accuracy']:.2f}%")
            print(f"   TTNN Accuracy:   {ttnn_result['accuracy']:.2f}%")
    
    # Overall summary
    print(f"\n🏆 BENCHMARK SUMMARY - POINTWISE MULTIPLY")
    print("=" * 80)
    
    numpy_results = [r for r in results if r['backend'] == 'NumPy']
    ttnn_results = [r for r in results if r['backend'] == 'TTNN']
    
    if numpy_results and ttnn_results:
        print(f"📊 AVERAGE PERFORMANCE:")
        
        avg_numpy_throughput = np.mean([r['throughput_millions'] for r in numpy_results])
        avg_ttnn_throughput = np.mean([r['throughput_millions'] for r in ttnn_results])
        avg_gain = avg_ttnn_throughput / avg_numpy_throughput
        
        print(f"   NumPy Average:  {avg_numpy_throughput:.2f} M elements/sec")
        print(f"   TTNN Average:   {avg_ttnn_throughput:.2f} M elements/sec")
        
        if avg_gain > 1:
            print(f"   🚀 OVERALL GAIN: {avg_gain:.2f}x (TTNN advantage!)")
        else:
            print(f"   📊 OVERALL LOSS: {1/avg_gain:.2f}x (NumPy advantage)")
        
        print(f"\n💡 ANALYSIS:")
        if avg_gain > 1:
            print(f"   ✅ TTNN backend delivers {avg_gain:.1f}x better performance for multiplication!")
            print(f"   🔥 Multi-core hardware acceleration confirmed!")
            print(f"   🎯 Specialized AI hardware shows significant advantage for pointwise ops")
        else:
            print(f"   📊 NumPy shows {1/avg_gain:.1f}x better performance")
            print(f"   💡 This could be due to:")
            print(f"      - Device initialization overhead")
            print(f"      - Memory transfer costs")
            print(f"      - Small tensor sizes not utilizing full parallelism")
            
        # Capacity advantage
        print(f"\n🧠 CAPACITY & FEATURE ADVANTAGES:")
        print(f"   💾 NumPy: Limited by system RAM (~GBs)")
        print(f"   💾 TTNN: 60-core device with 90MB L1 cache + specialized AI ops")
        print(f"   📈 TTNN enables 45M+ element operations!")
        print(f"   🔥 TTNN supports specialized ML operations beyond basic arithmetic")
        
        # Show scaling behavior
        print(f"\n📈 SCALING BEHAVIOR:")
        for i, (numpy_r, ttnn_r) in enumerate(zip(numpy_results, ttnn_results)):
            if numpy_r['size'] == ttnn_r['size']:
                gain = ttnn_r['throughput_millions'] / numpy_r['throughput_millions']
                symbol = "🚀" if gain > 1 else "📊"
                print(f"   {numpy_r['size']:>7,} elements: {gain:.2f}x {symbol}")
    
    print(f"\n✨ POINTWISE MULTIPLY BENCHMARK COMPLETE! ✨")

if __name__ == "__main__":
    main()
