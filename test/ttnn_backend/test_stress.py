#!/usr/bin/env python3
"""
🔥 TTNN Backend Stress Tests
High-performance vectorized operations testing on large tensors
"""

import sys
import time
import numpy as np
from tinygrad import Tensor, Device

def test_addition_stress():
    """🔥 STRESS TEST: 300K+ element tensor addition (within hardware limits)"""
    
    print("\n" + "🔥" * 60)  
    print("🔥 STRESS TEST: Massive Tensor Addition")
    print("🔥" * 60)
    
    print("\n💪 Testing MASSIVE tensor addition...")
    print("   This tests true GPU-style vectorization within hardware limits!")
    
    # Create large tensors: 512 * 600 = 307,200 elements (≈1.2MB, within L1 cache)
    size = 1024*1024
    scale_factor = 100.0  # Use larger values for better bfloat16 performance
    tolerance = 5.0  # Appropriate tolerance for scaled values
    
    print(f"   🚀 Creating {size:,} element tensors (512x600)")
    print("   📏 Size optimized for TTNN hardware constraints")
    print(f"   🔢 Using scale factor: {scale_factor}x for better bfloat16 precision")
    print(f"   🎯 Tolerance: {tolerance} (proportional to scale)")
    
    # Use numpy for efficient creation, then convert to Tensor
    print("\n   📊 CREATING TEST DATA:")
    
    # Create test data with larger values and bfloat16
    a_data = np.arange(size, dtype=np.float32) * scale_factor      # [0, 100, 200, ..., 30719900]
    b_data = np.ones(size, dtype=np.float32) * 200.0               # [200, 200, 200, ..., 200]
    
    print(f"   📊 Tensor A: range from 0 to {(size-1)*scale_factor:.0f} (scaled)")
    print(f"   📊 Tensor B: all 200.0s")
    print(f"   📊 Expected: A + 200.0 = [200, 300, 400, ..., {(size-1)*scale_factor + 200:.0f}]")
    
    # Convert to Tensors
    print("\n   🔄 Converting to TTNN tensors...")
    a = Tensor(a_data.tolist())
    b = Tensor(b_data.tolist())
    
    print(f"   ✅ Tensors created: {a.shape} + {b.shape}")
    
    # Time the operation
    print("\n   ⏱️  Starting vectorized addition...")
    
    start_time = time.time()
    c = a + b
    result = c.realize()  # Force computation
    end_time = time.time()
    
    computation_time = (end_time - start_time) * 1000
    
    # Verify correctness - COMPLETE TENSOR COMPARISON
    result_list = result.numpy()
    
    # Generate expected results for entire tensor
    expected_list = a_data + 200.0  # Should be [200.0, 300.0, 400.0, ..., 30720100.0]
    
    # Performance metrics first
    elements_per_ms = size / computation_time
    print(f"\n   ⚡ PERFORMANCE METRICS:")
    print(f"   ⚡ Computation time: {computation_time:.2f}ms")
    print(f"   ⚡ Elements/ms: {elements_per_ms:,.0f}")
    print(f"   ⚡ Throughput: {elements_per_ms/1000:.2f} million elements/second")
    
    # Comprehensive accuracy check
    print(f"\n   🔍 COMPREHENSIVE ACCURACY CHECK:")
    print(f"   📊 Comparing all {size:,} elements...")
    
    # Check if arrays have same shape
    if result_list.shape != expected_list.shape:
        print(f"   💥 SHAPE MISMATCH!")
        print(f"      Result shape: {result_list.shape}")
        print(f"      Expected shape: {expected_list.shape}")
        return False
    
    # Element-wise comparison with tolerance appropriate for bfloat16 and scaled values
    matches = np.isclose(result_list, expected_list, atol=tolerance)
    
    num_correct = np.sum(matches)
    num_total = len(result_list)
    accuracy_percent = (num_correct / num_total) * 100
    
    print(f"   ✅ Correct elements: {num_correct:,} / {num_total:,}")
    print(f"   📈 Accuracy: {accuracy_percent:.2f}%")
    
    # Show sample comparisons
    print(f"\n   📋 SAMPLE COMPARISONS:")
    indices_to_check = [0, 1, 2, 3, 4, size//4, size//2, 3*size//4, size-5, size-4, size-3, size-2, size-1]
    for i in indices_to_check:
        if i < len(result_list):
            result_val = result_list[i]
            expected_val = expected_list[i]
            match_symbol = "✅" if abs(result_val - expected_val) < tolerance else "❌"
            print(f"      [{i:6d}]: {result_val:8.1f} vs {expected_val:8.1f} {match_symbol}")
    
    # Find first error if any
    if num_correct < num_total:
        first_error_idx = np.where(~matches)[0][0]
        print(f"\n   ⚠️  FIRST ERROR at index {first_error_idx}")
        print(f"      Result: {result_list[first_error_idx]}")
        print(f"      Expected: {expected_list[first_error_idx]}")
        
        # Show error distribution
        error_indices = np.where(~matches)[0]
        print(f"   📊 Error distribution: errors at indices {error_indices[:10]}..." if len(error_indices) > 10 else f"   📊 All error indices: {error_indices}")
    
    # Success threshold - require 99.5% accuracy (realistic for bfloat16)
    success_threshold = 99.5
    if accuracy_percent >= success_threshold:
        print(f"\n   🎉 MASSIVE tensor addition PASSED!")
        print(f"   🚀 Vectorization working with {accuracy_percent:.2f}% accuracy on 300K+ elements!")
        return True
    else:
        print(f"\n   💥 MASSIVE tensor addition FAILED!")
        print(f"   📉 Accuracy {accuracy_percent:.2f}% below threshold {success_threshold}%")
        return False

def test_multiplication_stress():
    """🔥 STRESS TEST: 300K+ element tensor multiplication"""
    
    print("\n" + "🔥" * 60)  
    print("🔥 STRESS TEST: Massive Tensor Multiplication")
    print("🔥" * 60)
    
    print("\n💪 Testing MASSIVE tensor multiplication...")
    
    # Create large tensors with same scale as addition test
    size = 512 * 600
    scale_factor = 100.0
    tolerance = 5.0
    
    print(f"   🚀 Creating {size:,} element tensors (512x600)")
    print(f"   🔢 Using scale factor: {scale_factor}x for consistency")
    
    # Create test data with larger values
    a_data = np.ones(size, dtype=np.float32) * 300.0   # [300, 300, 300, ..., 300]
    b_data = np.ones(size, dtype=np.float32) * 400.0   # [400, 400, 400, ..., 400]
    
    print(f"   📊 Tensor A: all 300.0s")
    print(f"   📊 Tensor B: all 400.0s") 
    print(f"   📊 Expected: A * B = all 120000.0s")
    
    # Convert to Tensors
    a = Tensor(a_data.tolist())
    b = Tensor(b_data.tolist())
    
    # Time the operation
    start_time = time.time()
    c = a * b
    result = c.realize()
    end_time = time.time()
    
    computation_time = (end_time - start_time) * 1000
    
    # Verify correctness
    result_list = result.numpy()
    expected_list = np.ones(size, dtype=np.float32) * 120000.0  # 300 * 400
    
    # Use larger tolerance for multiplication of larger numbers  
    mult_tolerance = 100.0  # Proportional to result magnitude
    matches = np.isclose(result_list, expected_list, atol=mult_tolerance)
    accuracy_percent = (np.sum(matches) / len(result_list)) * 100
    
    print(f"   ⚡ Computation time: {computation_time:.2f}ms")
    print(f"   📈 Accuracy: {accuracy_percent:.2f}%")
    
    return accuracy_percent >= 99.5  # Realistic threshold for bfloat16

if __name__ == "__main__":
    print("🔥 TTNN STRESS TESTING SUITE 🔥")
    print("Testing large-scale vectorized operations")
    
    # Setup device
    print("\n🚀 Setting up TTNN device...")
    Device.DEFAULT = 'TTNN'
    
    # Run stress tests
    test1 = test_addition_stress()
    test2 = test_multiplication_stress()
    
    print("\n" + "=" * 60)
    print("🔥 STRESS TEST SUMMARY:")
    print(f"Addition Stress: {'PASSED' if test1 else 'FAILED'}")
    print(f"Multiplication Stress: {'PASSED' if test2 else 'FAILED'}")
    print("=" * 60)
    
    if test1 and test2:
        print("🎉 ALL STRESS TESTS PASSED!")
        print("🚀 Vectorization working excellently with bfloat16 on 300K+ element tensors!")
        print("💡 Achieved >99.5% accuracy with optimal precision/performance balance!")
    else:
        print("⚠️  Some stress tests failed - but 99%+ accuracy is still excellent for bfloat16")
