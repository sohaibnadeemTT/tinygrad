#!/usr/bin/env python3
"""
Test to validate device copy flow in TTNN backend
"""

from tinygrad import Tensor, Device

def test_device_copy_flow():
    """Test explicit device copy operations"""
    
    print("=" * 60)
    print("Testing TTNN Device Copy Flow")
    print("=" * 60)
    
    print(f"Current backend: {Device.DEFAULT}")
    
    # Step 1: Create tensor (should be lazy, no device copy yet)
    print("\n1. Creating tensor (lazy, no device copy)...")
    a = Tensor([1.0, 2.0, 3.0])
    print(f"   Tensor created: device={a.device}")
    print(f"   Tensor is_realized: {a.uop.is_realized}")  # Should be False
    
    # Step 2: Force realization without computation (just copy to device)
    print("\n2. Forcing realization (copy to device)...")
    a_realized = a.realize()
    print(f"   Tensor is_realized: {a_realized.uop.is_realized}")  # Should be True
    print(f"   Buffer allocated: {a_realized.uop.buffer.is_allocated()}")
    
    # Step 3: Copy back to host and verify data
    print("\n3. Copying back to host...")
    result = a_realized.tolist()
    print(f"   Data copied back: {result}")
    
    # Step 4: Verify data integrity  
    expected = [1.0, 2.0, 3.0]
    print(f"   Expected: {expected}")
    
    if result == expected:
        print("   ✓ Device copy flow PASSED!")
        success = True
    else:
        print("   ✗ Device copy flow FAILED!")
        success = False
    
    # Step 5: Test with computation
    print("\n4. Testing with computation...")
    b = Tensor([4.0, 5.0, 6.0])
    print(f"   Created b: {b.tolist()}")
    
    # This should trigger: 
    # 1. Copy a to device (if not already)
    # 2. Copy b to device  
    # 3. Run addition kernel on device
    # 4. Copy result back to host
    c = a + b
    print(f"   Computed a + b, device={c.device}")
    print(f"   Is computed tensor realized: {c.uop.is_realized}")
    
    result_computed = c.tolist()
    expected_computed = [5.0, 7.0, 9.0]
    print(f"   Result: {result_computed}")
    print(f"   Expected: {expected_computed}")
    
    if result_computed == expected_computed:
        print("   ✓ Computation with device copy PASSED!")
        success = success and True
    else:
        print("   ✗ Computation with device copy FAILED!")
        success = False
    
    return success

def test_buffer_lifecycle():
    """Test the detailed buffer allocation/deallocation lifecycle"""
    
    print("\n" + "=" * 60)
    print("Testing Buffer Lifecycle")
    print("=" * 60)
    
    # Enable debug to see buffer operations
    import os
    old_debug = os.environ.get('DEBUG', '0')
    os.environ['DEBUG'] = '1'
    
    try:
        print("\n1. Creating tensor...")
        x = Tensor([10.0, 20.0, 30.0])
        print(f"   Before realize - is_realized: {x.uop.is_realized}")
        
        print("\n2. Accessing data (forces realization)...")
        data = x.tolist()
        print(f"   After tolist - is_realized: {x.uop.is_realized}")
        print(f"   Data: {data}")
        
        print("\n3. Creating second tensor and computing...")
        y = Tensor([1.0, 2.0, 3.0])
        z = x * y  # Element-wise multiplication
        
        print(f"   Before accessing result - z.is_realized: {z.uop.is_realized}")
        result = z.tolist()
        print(f"   After accessing result - z.is_realized: {z.uop.is_realized}")
        print(f"   Result: {result}")
        
        expected = [10.0, 40.0, 90.0]  # [10*1, 20*2, 30*3]
        print(f"   Expected: {expected}")
        
        if result == expected:
            print("   ✓ Buffer lifecycle test PASSED!")
            return True
        else:
            print("   ✗ Buffer lifecycle test FAILED!")
            return False
            
    finally:
        # Restore debug level
        os.environ['DEBUG'] = old_debug

if __name__ == "__main__":
    print("Testing TTNN backend device copy validation...")
    
    # Test 1: Basic device copy flow
    test1_result = test_device_copy_flow()
    
    # Test 2: Buffer lifecycle  
    #test2_result = test_buffer_lifecycle()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Device Copy Flow: {'PASSED' if test1_result else 'FAILED'}")
    print(f"Buffer Lifecycle: {'PASSED' if test2_result else 'FAILED'}")
    print("=" * 60)
    
    if test1_result and test2_result:
        print("🎉 All device copy tests PASSED!")
    else:
        print("❌ Some device copy tests FAILED!")
