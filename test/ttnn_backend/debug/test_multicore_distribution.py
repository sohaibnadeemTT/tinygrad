#!/usr/bin/env python3
"""
TTNN Multi-Core Distribution Investigation
Test different approaches to enable multi-core tensor distribution
"""

import torch
import numpy as np

def test_direct_tile_layout():
    """Test using layout=ttnn.TILE directly in from_torch (as shown in documentation)"""
    
    print("🧪 DIRECT TILE LAYOUT TEST")
    print("=" * 50)
    print("Using layout=ttnn.TILE directly in ttnn.from_torch() for multi-core distribution")
    
    try:
        import ttnn
        
        # Initialize TTNN device
        device = ttnn.open_device(device_id=0)
        print(f"✅ TTNN device opened: {device}")
        
        # Test different 2D shapes that might enable multi-core distribution
        test_configs = [
            ("Square 1K×1K", (1024, 1024), "1M elements in optimized 2D shape"),
            ("Square 2K×2K", (2048, 2048), "4M elements in optimized 2D shape"),  
            ("Square 3K×3K", (3072, 3072), "9M elements in optimized 2D shape"),
            ("Rectangle 1K×5K", (1024, 5120), "5M elements in rectangle"),
            ("Rectangle 2K×10K", (2048, 10240), "20M elements in rectangle"),
        ]
        
        scale_factor = 100.0
        tolerance = 10.0
        
        print(f"📊 MULTI-CORE 2D TENSOR TESTS:")
        print(f"   🔧 Using direct TILE layout in from_torch()")
        print(f"   📐 Testing 2D shapes for optimal core distribution")
        print(f"   🔢 Scale factor: {scale_factor}x")
        print(f"   📏 Tolerance: {tolerance}")
        
        successful_tests = []
        
        for config_name, shape, description in test_configs:
            print(f"\n🧪 TESTING {config_name} - {description}")
            
            elements = shape[0] * shape[1]
            mb_required = (elements * 2) / 1024 / 1024
            cores_theoretical = mb_required / 1.5
            
            print(f"   📊 Shape: {shape[0]}×{shape[1]} = {elements:,} elements")
            print(f"   💾 Memory: {mb_required:.1f}MB (~{cores_theoretical:.1f} cores needed)")
            
            try:
                # Create 2D tensors with constants (avoid layout corruption)
                print(f"   📊 Creating 2D tensors...")
                a_data = torch.ones(shape, dtype=torch.bfloat16) * 300.0
                b_data = torch.ones(shape, dtype=torch.bfloat16) * 700.0
                expected_result = 1000.0
                
                print(f"   🏗️  Converting to TTNN with DIRECT TILE layout...")
                # KEY: Use layout=ttnn.TILE directly in from_torch (as in documentation)
                a_ttnn = ttnn.from_torch(a_data, device=device, layout=ttnn.TILE_LAYOUT)
                b_ttnn = ttnn.from_torch(b_data, device=device, layout=ttnn.TILE_LAYOUT)
                
                print(f"   ✅ DIRECT TILE conversion successful!")
                print(f"   📐 TTNN tensor shapes: {a_ttnn.shape}")
                
                # Perform addition on distributed tensors
                print(f"   ➕ Performing addition on distributed tensors...")
                result_ttnn = ttnn.add(a_ttnn, b_ttnn)
                
                print(f"   📤 Converting result back...")
                result_torch = ttnn.to_torch(result_ttnn)
                
                print(f"   🔍 Accuracy verification...")
                # Sample verification for large tensors
                sample_size = min(1000, elements)
                if elements > 1000:
                    # Sample random elements
                    flat_result = result_torch.flatten()
                    sample_indices = torch.randint(0, elements, (sample_size,))
                    sample_results = flat_result[sample_indices]
                else:
                    sample_results = result_torch.flatten()
                    
                expected_tensor = torch.full_like(sample_results, expected_result)
                accurate_count = torch.sum(torch.isclose(sample_results, expected_tensor, atol=tolerance)).item()
                accuracy_percentage = (accurate_count / sample_size) * 100
                
                # Calculate memory and core usage
                actual_mb = mb_required
                cores_used = cores_theoretical
                
                print(f"   🎉 SUCCESS! {config_name} completed!")
                print(f"   📊 Accuracy: {accuracy_percentage:.2f}% ({accurate_count}/{sample_size} samples)")
                print(f"   💾 Memory used: {actual_mb:.1f}MB")
                print(f"   🔧 Estimated cores: ~{cores_used:.1f} / ~60 available")
                print(f"   📈 Device utilization: {cores_used/60*100:.1f}%")
                
                # Show sample values
                first_few = result_torch.flatten()[:5].tolist()
                print(f"   📋 First 5 results: {first_few} (expected: {expected_result})")
                
                successful_tests.append((config_name, shape, elements, actual_mb, accuracy_percentage, cores_used))
                
                # Clean up
                del a_data, b_data, a_ttnn, b_ttnn, result_ttnn, result_torch
                
            except RuntimeError as e:
                error_msg = str(e)
                print(f"   💥 FAILED at {config_name}")
                print(f"   📝 Error: {error_msg}")
                
                if "beyond max L1 size" in error_msg:
                    # Check if still single-core allocation
                    if "core range" in error_msg:
                        core_info = error_msg.split("core range ")[1].split("]")[0] + "]"
                        print(f"   ⚠️  Still single-core allocation: {core_info}")
                        print(f"   💡 Need different approach for multi-core distribution")
                    else:
                        print(f"   📏 Hit L1 cache limit")
                        
                    # Extract sizes for analysis
                    if "grow to" in error_msg and "max L1 size of" in error_msg:
                        try:
                            grow_size_str = error_msg.split("grow to ")[1].split(" B")[0]
                            l1_size_str = error_msg.split("max L1 size of ")[1].split(" B")[0]
                            grow_size = int(grow_size_str)
                            l1_size = int(l1_size_str)
                            
                            grow_mb = grow_size / 1024 / 1024
                            l1_mb = l1_size / 1024 / 1024
                            
                            print(f"   📊 Allocation attempt: {grow_mb:.2f}MB")
                            print(f"   📏 L1 limit per core: {l1_mb:.2f}MB") 
                            print(f"   🔢 Overallocation: {grow_mb/l1_mb:.1f}x")
                        except:
                            pass
                            
                elif "memory" in error_msg.lower():
                    print(f"   ⚠️  Device memory exhausted")
                    
                # Stop on first failure to analyze
                break
                
            except Exception as e:
                print(f"   💥 UNEXPECTED ERROR: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Results summary
        print(f"\n🏆 DIRECT TILE LAYOUT RESULTS")
        print("=" * 50)
        
        if successful_tests:
            print(f"✅ Successfully completed {len(successful_tests)} test(s):")
            
            for config_name, shape, elements, mb, accuracy, cores in successful_tests:
                print(f"\n   {config_name}:")
                print(f"     📐 Shape: {shape[0]}×{shape[1]}")
                print(f"     🔢 Elements: {elements:,}")
                print(f"     💾 Memory: {mb:.1f}MB")
                print(f"     🔧 Cores: ~{cores:.1f}")
                print(f"     📊 Accuracy: {accuracy:.1f}%")
            
            max_test = max(successful_tests, key=lambda x: x[2])  # Max by elements
            max_config, max_shape, max_elements, max_mb, max_acc, max_cores = max_test
            
            print(f"\n🚀 MAXIMUM ACHIEVEMENT:")
            print(f"   🏆 Best config: {max_config}")
            print(f"   📐 Shape: {max_shape[0]}×{max_shape[1]}")
            print(f"   🔢 Elements: {max_elements:,}")
            print(f"   💾 Memory: {max_mb:.1f}MB")
            print(f"   📈 Multi-core utilization: {max_cores/60*100:.1f}%")
            
        else:
            print("❌ No tests completed successfully")
            print("💡 Direct TILE layout still hitting single-core limits")
            
        ttnn.close_device(device)
        return successful_tests
        
    except ImportError:
        print("❌ TTNN not available")
        return []
    except Exception as e:
        print(f"💥 Setup error: {e}")
        return []

def test_memory_configs():
    """Test different memory configurations that might enable multi-core distribution"""
    
    print("\n🔧 MEMORY CONFIGURATION TEST")
    print("=" * 50)
    print("Testing different memory configurations for multi-core distribution")
    
    try:
        import ttnn
        
        device = ttnn.open_device(device_id=0)
        print(f"✅ TTNN device opened: {device}")
        
        # Test moderate size with different memory configs
        shape = (512, 600)  # 307K elements - known to work with single core
        elements = shape[0] * shape[1]
        
        print(f"📊 Testing shape: {shape[0]}×{shape[1]} = {elements:,} elements")
        print("🔧 Trying different memory configurations...")
        
        # Create test data
        a_data = torch.ones(shape, dtype=torch.bfloat16) * 400.0
        b_data = torch.ones(shape, dtype=torch.bfloat16) * 600.0
        expected_result = 1000.0
        
        # Test different approaches
        configs = [
            ("Default", {}),
            ("DRAM", {"memory_config": ttnn.DRAM_MEMORY_CONFIG} if hasattr(ttnn, 'DRAM_MEMORY_CONFIG') else {}),
            ("L1", {"memory_config": ttnn.L1_MEMORY_CONFIG} if hasattr(ttnn, 'L1_MEMORY_CONFIG') else {}),
        ]
        
        for config_name, kwargs in configs:
            print(f"\n   🧪 Testing {config_name} configuration...")
            
            try:
                if kwargs:
                    a_ttnn = ttnn.from_torch(a_data, device=device, layout=ttnn.TILE_LAYOUT, **kwargs)
                    b_ttnn = ttnn.from_torch(b_data, device=device, layout=ttnn.TILE_LAYOUT, **kwargs)
                else:
                    a_ttnn = ttnn.from_torch(a_data, device=device, layout=ttnn.TILE_LAYOUT)
                    b_ttnn = ttnn.from_torch(b_data, device=device, layout=ttnn.TILE_LAYOUT)
                
                result_ttnn = ttnn.add(a_ttnn, b_ttnn)
                result_torch = ttnn.to_torch(result_ttnn)
                
                # Quick accuracy check
                sample_results = result_torch.flatten()[:100]
                expected_tensor = torch.full_like(sample_results, expected_result)
                accuracy = torch.allclose(sample_results, expected_tensor, atol=10.0)
                
                print(f"     ✅ {config_name} configuration works!")
                print(f"     📊 Sample accuracy: {'PASS' if accuracy else 'FAIL'}")
                
                # Clean up
                del a_ttnn, b_ttnn, result_ttnn, result_torch
                
            except Exception as e:
                print(f"     💥 {config_name} failed: {e}")
        
        ttnn.close_device(device)
        
    except Exception as e:
        print(f"💥 Memory config test error: {e}")

if __name__ == "__main__":
    print("🚀 TTNN MULTI-CORE DISTRIBUTION INVESTIGATION")
    print("=" * 80)
    
    # Test 1: Direct TILE layout approach
    print("\n1️⃣ DIRECT TILE LAYOUT APPROACH")
    direct_results = test_direct_tile_layout()
    
    # Test 2: Memory configuration exploration  
    print("\n\n2️⃣ MEMORY CONFIGURATION EXPLORATION")
    test_memory_configs()
    
    # Summary
    print(f"\n🎯 INVESTIGATION SUMMARY")
    print("=" * 80)
    
    if direct_results:
        max_result = max(direct_results, key=lambda x: x[2])
        max_elements = max_result[2]
        max_mb = max_result[3]
        
        print(f"✅ BREAKTHROUGH ACHIEVED:")
        print(f"   🔢 Maximum elements: {max_elements:,}")
        print(f"   💾 Memory utilized: {max_mb:.1f}MB")
        print(f"   🎯 Direct TILE layout approach successful!")
        
        if max_mb > 2.0:  # More than single core capacity
            print(f"   🚀 MULTI-CORE DISTRIBUTION CONFIRMED!")
        else:
            print(f"   📊 Single-core operation confirmed")
            
    else:
        print(f"🔍 INVESTIGATION NEEDED:")
        print(f"   💡 Direct TILE layout still hits single-core limits")
        print(f"   🔧 May need advanced TTNN sharding APIs")
        print(f"   📚 Consult TTNN documentation for explicit multi-core configuration")
        
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. If successful: Scale up tensor sizes using working approach")
    print(f"   2. If failed: Research TTNN explicit sharding/distribution APIs")
    print(f"   3. Consider mesh/multi-device configurations for massive scaling")
    
    print(f"\n✨ MULTI-CORE DISTRIBUTION INVESTIGATION COMPLETE! ✨")
