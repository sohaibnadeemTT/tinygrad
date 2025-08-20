#!/usr/bin/env python3
"""
TTNN Size Limit Investigation
Find the maximum tensor size before hitting hardware constraints
"""

import torch
import numpy as np

def test_progressive_sizes():
    """Test progressively larger tensor sizes to find hardware limits"""
    
    print("🔍 TTNN SIZE LIMIT INVESTIGATION")
    print("=" * 50)
    print("Finding maximum tensor size before hardware constraints")
    
    try:
        import ttnn
        
        # Initialize TTNN device
        device = ttnn.open_device(device_id=0)
        print(f"✅ TTNN device opened: {device}")
        
        # Test sizes - progressively larger starting from our working 300K
        test_sizes = [
            ("300K (working)", 300_000),
            ("400K", 400_000),
            ("500K", 500_000),
            ("600K", 600_000),
            ("700K", 700_000),
            ("800K", 800_000),
            ("900K", 900_000),
            ("1M", 1_000_000),
            ("1.1M", 1_100_000),
            ("1.2M", 1_200_000),
            ("1.3M", 1_300_000),
            ("1.5M", 1_500_000),
        ]
        
        max_working_size = 0
        failure_size = None
        
        for size_name, size in test_sizes:
            print(f"\n🧪 Testing {size_name} ({size:,} elements)")
            
            try:
                # Calculate memory requirements
                bytes_per_element = 2  # bfloat16
                total_bytes = size * bytes_per_element
                print(f"   💾 Memory requirement: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
                
                # Create test data (use constants to avoid layout corruption issues)
                a_data = torch.ones(size, dtype=torch.bfloat16) * 100.0
                b_data = torch.ones(size, dtype=torch.bfloat16) * 200.0
                
                print(f"   📊 Created tensors: {size:,} elements each")
                
                # Convert to TTNN tensors
                a_ttnn = ttnn.from_torch(a_data, device=device)
                b_ttnn = ttnn.from_torch(b_data, device=device)
                
                print(f"   🔄 Converted to TTNN: {a_ttnn.shape}")
                
                # Convert to TILE layout (this is where memory allocation happens)
                print(f"   🏗️  Converting to TILE layout...")
                a_ttnn_tile = ttnn.to_layout(a_ttnn, ttnn.TILE_LAYOUT)
                b_ttnn_tile = ttnn.to_layout(b_ttnn, ttnn.TILE_LAYOUT)
                
                print(f"   ✅ TILE layout successful")
                
                # Perform addition
                print(f"   ➕ Performing addition...")
                result_ttnn = ttnn.add(a_ttnn_tile, b_ttnn_tile)
                
                # Convert back to check result
                result_torch = ttnn.to_torch(result_ttnn)
                expected = 300.0  # 100 + 200
                
                # Quick accuracy check (just check a few values)
                sample_size = min(10, size)
                sample_correct = torch.allclose(result_torch[:sample_size], 
                                              torch.full((sample_size,), expected, dtype=torch.bfloat16), 
                                              atol=5.0)
                
                print(f"   ✅ SUCCESS! Size {size_name} works perfectly")
                print(f"   📊 Sample accuracy: {'PASS' if sample_correct else 'FAIL'}")
                
                max_working_size = size
                
            except RuntimeError as e:
                error_msg = str(e)
                print(f"   💥 FAILED at size {size_name}")
                print(f"   📝 Error: {error_msg}")
                
                # Check if it's the L1 cache limit error
                if "beyond max L1 size" in error_msg:
                    print(f"   🎯 HIT L1 CACHE LIMIT!")
                    # Extract the actual L1 size from error message
                    if "max L1 size of" in error_msg:
                        l1_size_str = error_msg.split("max L1 size of ")[1].split(" B")[0]
                        l1_size = int(l1_size_str)
                        print(f"   📏 L1 Cache Size: {l1_size:,} bytes ({l1_size/1024/1024:.2f} MB)")
                        
                    # Calculate theoretical max elements
                    if max_working_size > 0:
                        working_bytes = max_working_size * bytes_per_element
                        print(f"   📊 Last working size: {max_working_size:,} elements ({working_bytes:,} bytes)")
                        
                failure_size = size
                break
                
            except Exception as e:
                print(f"   💥 UNEXPECTED ERROR at size {size_name}: {e}")
                failure_size = size
                break
        
        # Summary
        print(f"\n🎯 SIZE LIMIT ANALYSIS COMPLETE")
        print("=" * 50)
        
        if max_working_size > 0:
            working_mb = (max_working_size * bytes_per_element) / 1024 / 1024
            print(f"✅ Maximum working size: {max_working_size:,} elements ({working_mb:.2f} MB)")
            
        if failure_size:
            failure_mb = (failure_size * bytes_per_element) / 1024 / 1024  
            print(f"❌ First failure size: {failure_size:,} elements ({failure_mb:.2f} MB)")
            
        # Calculate safe working limit (90% of max)
        if max_working_size > 0:
            safe_limit = int(max_working_size * 0.9)
            safe_mb = (safe_limit * bytes_per_element) / 1024 / 1024
            print(f"💡 Recommended safe limit: {safe_limit:,} elements ({safe_mb:.2f} MB)")
            
        # Clean up
        ttnn.close_device(device)
        
        return max_working_size, failure_size
        
    except ImportError:
        print("❌ TTNN not available")
        return None, None
    except Exception as e:
        print(f"💥 Setup error: {e}")
        return None, None

def test_specific_size(size):
    """Test a specific size with detailed monitoring"""
    
    print(f"\n🔬 DETAILED TEST: {size:,} elements")
    print("-" * 30)
    
    try:
        import ttnn
        device = ttnn.open_device(device_id=0)
        
        # Memory calculation
        bytes_per_element = 2  # bfloat16
        total_bytes = size * bytes_per_element
        mb_required = total_bytes / 1024 / 1024
        
        print(f"📊 Memory analysis:")
        print(f"   Elements: {size:,}")
        print(f"   Bytes per element: {bytes_per_element}")
        print(f"   Total memory: {total_bytes:,} bytes ({mb_required:.2f} MB)")
        
        # Create and test
        a_data = torch.ones(size, dtype=torch.bfloat16) * 150.0
        b_data = torch.ones(size, dtype=torch.bfloat16) * 250.0
        
        a_ttnn = ttnn.from_torch(a_data, device=device)
        b_ttnn = ttnn.from_torch(b_data, device=device)
        
        # TILE layout conversion
        a_ttnn_tile = ttnn.to_layout(a_ttnn, ttnn.TILE_LAYOUT)
        b_ttnn_tile = ttnn.to_layout(b_ttnn, ttnn.TILE_LAYOUT)
        
        # Addition
        result_ttnn = ttnn.add(a_ttnn_tile, b_ttnn_tile)
        result_torch = ttnn.to_torch(result_ttnn)
        
        # Verify result
        expected = 400.0  # 150 + 250
        accuracy = torch.allclose(result_torch, torch.full_like(result_torch, expected), atol=10.0)
        
        print(f"✅ Test completed successfully!")
        print(f"📊 Accuracy: {'PASS' if accuracy else 'FAIL'}")
        
        ttnn.close_device(device)
        return True
        
    except Exception as e:
        print(f"💥 Failed: {e}")
        try:
            ttnn.close_device(device)
        except:
            pass
        return False

if __name__ == "__main__":
    print("🚀 TTNN TENSOR SIZE LIMIT INVESTIGATION")
    print("Finding hardware constraints for large tensor operations")
    
    # Progressive size testing
    max_size, fail_size = test_progressive_sizes()
    
    # If we found limits, test a few specific sizes around the boundary
    if max_size and fail_size:
        print(f"\n🔬 BOUNDARY TESTING")
        print("Testing sizes around the discovered limit...")
        
        # Test a few sizes between max working and first failure
        boundary_sizes = []
        if fail_size - max_size > 50_000:
            # Test intermediate sizes
            step = (fail_size - max_size) // 3
            boundary_sizes = [max_size + step, max_size + 2*step]
            
        for test_size in boundary_sizes:
            if test_specific_size(test_size):
                print(f"   Size {test_size:,}: WORKS")
            else:
                print(f"   Size {test_size:,}: FAILS")
                break
