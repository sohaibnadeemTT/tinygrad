#!/usr/bin/env python3
"""
Test addition operations on TTNN backend
"""

from tinygrad import Tensor, Device

def test_simple_addition():
    """Test basic elementwise addition on TTNN"""
    
    print("=" * 60)
    print("Testing TTNN Addition Operations")
    print("=" * 60)
    
    print(f"Current backend: {Device.DEFAULT}")
    
    # Test 1: Simple addition
    print("\n1. Testing simple addition: [1,2,3] + [4,5,6]...")
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    
    print(f"   Tensor a: {a.tolist()}, device: {a.device}")
    print(f"   Tensor b: {b.tolist()}, device: {b.device}")
    
    # Perform addition
    c = a + b
    print(f"   Addition tensor c, device: {c.device}")
    print(f"   Is computed tensor realized: {c.uop.is_realized}")
    
    # Get result
    result = c.tolist()
    expected = [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]
    
    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    
    if result == expected:
        print("   ✓ Simple addition PASSED!")
        return True
    else:
        print("   ✗ Simple addition FAILED!")
        return False

def test_addition_with_different_values():
    """Test addition with different value ranges"""
    
    print("\n" + "=" * 60)
    print("Testing Addition with Different Values")
    print("=" * 60)
    
    test_cases = [
        # (a_values, b_values, expected_name)
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "ones addition"),
        ([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], "mixed scale"),
        ([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0], "negative numbers"),
        ([0.5, 1.5, 2.5], [0.1, 0.2, 0.3], "decimal values"),
    ]
    
    all_passed = True
    
    for i, (a_vals, b_vals, test_name) in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_name}...")
        
        a = Tensor(a_vals)
        b = Tensor(b_vals)
        
        expected = [a_val + b_val for a_val, b_val in zip(a_vals, b_vals)]
        result = (a + b).tolist()
        
        print(f"   {a_vals} + {b_vals} = {result}")
        print(f"   Expected: {expected}")
        
        if result == expected:
            print(f"   ✓ {test_name} PASSED!")
        else:
            print(f"   ✗ {test_name} FAILED!")
            all_passed = False
    
    return all_passed

def test_elementwise_multiplication():
    """Test elementwise multiplication (bonus test)"""
    
    print("\n" + "=" * 60)
    print("Testing Elementwise Multiplication")
    print("=" * 60)
    
    print("\n1. Testing elementwise multiplication: [10,20,30] * [1,2,3]...")
    
    x = Tensor([10.0, 20.0, 30.0])
    y = Tensor([1.0, 2.0, 3.0])
    
    print(f"   Tensor x: {x.tolist()}")
    print(f"   Tensor y: {y.tolist()}")
    
    # Element-wise multiplication
    z = x * y
    
    result = z.tolist()
    expected = [10.0, 40.0, 90.0]  # [10*1, 20*2, 30*3]
    
    print(f"   Result: {result}")
    print(f"   Expected: {expected}")
    
    if result == expected:
        print("   ✓ Elementwise multiplication PASSED!")
        return True
    else:
        print("   ✗ Elementwise multiplication FAILED!")
        return False

def test_addition_performance():
    """Test addition with larger tensors"""
    
    print("\n" + "=" * 60)  
    print("Testing Addition Performance")
    print("=" * 60)
    
    print("\n1. Testing larger tensor addition...")
    
    # Create larger tensors
    size = 10
    a_vals = [float(i) for i in range(size)]  # [0, 1, 2, ..., 9]
    b_vals = [1.0] * size                     # [1, 1, 1, ..., 1]
    
    a = Tensor(a_vals)
    b = Tensor(b_vals)
    
    print(f"   Testing {size}-element tensors")
    print(f"   a: {a_vals[:3]}...{a_vals[-3:]} (showing first/last 3)")
    print(f"   b: {b_vals[:3]}...{b_vals[-3:]} (showing first/last 3)")
    
    # Time the operation
    import time
    start_time = time.time()
    c = a + b
    result = c.tolist()
    end_time = time.time()
    
    expected = [a_val + 1.0 for a_val in a_vals]
    
    print(f"   Result: {result[:3]}...{result[-3:]} (showing first/last 3)")
    print(f"   Expected: {expected[:3]}...{expected[-3:]} (showing first/last 3)")
    print(f"   Time: {(end_time - start_time)*1000:.2f}ms")
    
    if result == expected:
        print("   ✓ Large tensor addition PASSED!")
        return True
    else:
        print("   ✗ Large tensor addition FAILED!")
        return False

if __name__ == "__main__":
    print("Testing TTNN backend addition operations...")
    
    # Run all tests
    test1 = test_simple_addition()
    test2 = test_addition_with_different_values()
    test3 = test_elementwise_multiplication()
    test4 = test_addition_performance()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Simple Addition: {'PASSED' if test1 else 'FAILED'}")
    print(f"Different Values: {'PASSED' if test2 else 'FAILED'}")
    print(f"Multiplication: {'PASSED' if test3 else 'FAILED'}")
    print(f"Performance: {'PASSED' if test4 else 'FAILED'}")
    print("=" * 60)
    
    all_passed = test1 and test2 and test3 and test4
    if all_passed:
        print("🎉 All addition tests PASSED!")
    else:
        print("❌ Some addition tests FAILED!")
        print("\nNote: Results of [0.0, 0.0, 0.0] indicate kernel computation issues.")
        print("This suggests the workitem functions need to be fixed to process all elements.")
