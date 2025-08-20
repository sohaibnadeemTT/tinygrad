#!/usr/bin/env python3
"""
Example usage of the new TTNN backend in tinygrad
"""

from tinygrad import Tensor, Device

def test_ttnn_backend():
    """Test basic elementwise operations with the TTNN backend"""
    
    print("=" * 50)
    print("Testing TTNN Backend - Simple Addition")
    print("=" * 50)
    
    # Show current default backend
    print(f"Current default backend: {Device.DEFAULT}")
    
    # Check if TTNN is available
    try:
        ttnn_device = Device["TTNN"]
        print(f"✓ TTNN backend is available: {ttnn_device}")
    except Exception as e:
        print(f"✗ TTNN backend failed to load: {e}")
        return False
    
    print("\n--- Testing Simple Elementwise Addition ---")
    
    try:
        # Create simple tensors on TTNN
        print("Creating tensors on TTNN...")
        a = Tensor([1.0, 2.0, 3.0])  # Should use TTNN as default backend
        b = Tensor([4.0, 5.0, 6.0])  # Should use TTNN as default backend
        
        print(f"Tensor a: {a.tolist()}, device: {a.device}")
        print(f"Tensor b: {b.tolist()}, device: {b.device}")
        
        # Perform elementwise addition
        print("Performing elementwise addition: a + b")
        c = a + b
        
        print(f"Tensor c device: {c.device}")
        
        # Copy result back to host and check
        print("Copying result back to host...")
        result = c.tolist()
        print(f"Result: {result}")
        
        # Verify the result
        expected = [5.0, 7.0, 9.0]  # [1+4, 2+5, 3+6]
        print(f"Expected: {expected}")
        
        if result == expected:
            print("✓ Elementwise addition PASSED!")
            return True
        else:
            print("✗ Elementwise addition FAILED - results don't match")
            return False
            
    except Exception as e:
        print(f"✗ Elementwise addition FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_available_backends():
    """Show all available backends"""
    print("\n--- Available Backends ---")
    
    from tinygrad.device import ALL_DEVICES
    print(f"All registered devices: {ALL_DEVICES}")
    
    print("\nTesting device availability:")
    for device_name in ALL_DEVICES:
        try:
            device = Device[device_name]
            print(f"✓ {device_name}: Available")
        except Exception as e:
            print(f"✗ {device_name}: {str(e)[:50]}...")

if __name__ == "__main__":
    show_available_backends()
    test_ttnn_backend()
    
    print("\n" + "=" * 50)
    print("TTNN backend testing complete!")
    print("=" * 50)
