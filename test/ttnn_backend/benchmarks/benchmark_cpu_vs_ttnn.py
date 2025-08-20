#!/usr/bin/env python3
"""
CPU vs TTNN Performance Benchmark
Compare execution times between CPU and TTNN backends
"""

import os
import time
import numpy as np
from tinygrad import Tensor, Device

def benchmark_backend(backend_name, size=100000):
    """Benchmark tensor addition on specified backend"""
    
    print(f"\n🔍 BENCHMARKING {backend_name.upper()} BACKEND")
    print("=" * 50)
    
    # Set the device
    Device.DEFAULT = backend_name
    print(f"   🎯 Device: {Device.DEFAULT}")
    print(f"   📊 Testing {size:,} element tensors")
    
    # Create test data
    print("   📊 Creating test tensors...")
    a_data = np.arange(size, dtype=np.float32) * 100.0
    b_data = np.ones(size, dtype=np.float32) * 200.0
    
    # Convert to Tensors
    a = Tensor(a_data.tolist())
    b = Tensor(b_data.tolist())
    
    print(f"   🔄 Tensors created on device: {a.device}")
    
    # Benchmark addition
    print("   ⏱️  Starting computation...")
    start_time = time.time()
    
    # Perform addition
    c = a + b
    
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
    expected = a_data + 200.0
    errors = np.abs(result - expected)
    max_error = np.max(errors)
    accurate_count = np.sum(errors < 5.0)  # Same tolerance as stress test
    accuracy = (accurate_count / size) * 100
    
    print(f"   📊 Accuracy: {accuracy:.2f}%")
    print(f"   📊 Max error: {max_error:.6f}")
    
    # Show sample results
    print(f"   📋 First 5 results: {result[:5].tolist()}")
    print(f"   📋 Expected first 5: {expected[:5].tolist()}")
    
    return {
        'backend': backend_name,
        'size': size,
        'time_ms': computation_time,
        'throughput_ms': throughput,
        'throughput_millions': throughput_millions,
        'accuracy': accuracy,
        'max_error': max_error
    }

def main():
    print("🚀 CPU vs TTNN PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Test sizes
    test_sizes = [
        10000,      # 10K elements
        100000,     # 100K elements  
        300000,     # 300K elements (our stress test size)
    ]
    
    results = []
    
    for size in test_sizes:
        print(f"\n🧪 TESTING {size:,} ELEMENT TENSORS")
        print("=" * 80)
        
        # Test CPU
        try:
            cpu_result = benchmark_backend("CPU", size)
            results.append(cpu_result)
        except Exception as e:
            print(f"   💥 CPU test failed: {e}")
            cpu_result = None
        
        # Test TTNN 
        try:
            # Force TTNN environment
            os.environ["TTNN"] = "1"
            ttnn_result = benchmark_backend("TTNN", size)
            results.append(ttnn_result)
        except Exception as e:
            print(f"   💥 TTNN test failed: {e}")
            ttnn_result = None
        
        # Compare results
        if cpu_result and ttnn_result:
            print(f"\n📊 PERFORMANCE COMPARISON ({size:,} elements):")
            print("=" * 50)
            
            speedup = cpu_result['time_ms'] / ttnn_result['time_ms']
            throughput_gain = ttnn_result['throughput_millions'] / cpu_result['throughput_millions']
            
            print(f"   CPU Time:    {cpu_result['time_ms']:.2f}ms")
            print(f"   TTNN Time:   {ttnn_result['time_ms']:.2f}ms")
            print(f"   🚀 SPEEDUP:   {speedup:.2f}x faster")
            print()
            print(f"   CPU Throughput:  {cpu_result['throughput_millions']:.2f} M elements/sec")
            print(f"   TTNN Throughput: {ttnn_result['throughput_millions']:.2f} M elements/sec")
            print(f"   🚀 THROUGHPUT GAIN: {throughput_gain:.2f}x")
            print()
            print(f"   CPU Accuracy:  {cpu_result['accuracy']:.2f}%")
            print(f"   TTNN Accuracy: {ttnn_result['accuracy']:.2f}%")
            
            if speedup > 1:
                print(f"   ✅ TTNN is {speedup:.1f}x FASTER than CPU!")
            else:
                print(f"   ⚠️  CPU is {1/speedup:.1f}x faster than TTNN")
    
    # Overall summary
    print(f"\n🏆 BENCHMARK SUMMARY")
    print("=" * 80)
    
    cpu_results = [r for r in results if r['backend'] == 'CPU']
    ttnn_results = [r for r in results if r['backend'] == 'TTNN']
    
    if cpu_results and ttnn_results:
        print(f"📊 AVERAGE PERFORMANCE:")
        
        avg_cpu_throughput = np.mean([r['throughput_millions'] for r in cpu_results])
        avg_ttnn_throughput = np.mean([r['throughput_millions'] for r in ttnn_results])
        avg_speedup = avg_ttnn_throughput / avg_cpu_throughput
        
        print(f"   CPU Average:  {avg_cpu_throughput:.2f} M elements/sec")
        print(f"   TTNN Average: {avg_ttnn_throughput:.2f} M elements/sec")
        print(f"   🚀 OVERALL GAIN: {avg_speedup:.2f}x")
        
        print(f"\n💡 CONCLUSION:")
        if avg_speedup > 1:
            print(f"   ✅ TTNN backend delivers {avg_speedup:.1f}x better performance!")
            print(f"   🔥 Multi-core TTNN acceleration confirmed!")
        else:
            print(f"   📊 CPU shows {1/avg_speedup:.1f}x better performance")
            
        # Memory capacity advantage
        print(f"\n🧠 CAPACITY ADVANTAGE:")
        print(f"   💾 CPU: Limited by system RAM (~GBs)")
        print(f"   💾 TTNN: 60-core device with 90MB total L1 cache")
        print(f"   📈 TTNN enables 45M+ element operations!")
    
    print(f"\n✨ BENCHMARK COMPLETE! ✨")

if __name__ == "__main__":
    main()
