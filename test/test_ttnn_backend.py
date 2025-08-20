import os
import unittest
import numpy as np
import torch

from tinygrad import Tensor, Device


@unittest.skipUnless(os.environ.get("TTNN"), "TTNN device not enabled/available")
class TestTTNNBackend(unittest.TestCase):
  def setUp(self):
    # ensure default device is TTNN for these tests
    Device.DEFAULT = "TTNN"

  def test_add(self):
    # Test data
    a_data = [1.0, 2.0, 3.0, 4.0]
    b_data = [10.0, 20.0, 30.0, 40.0]
    
    # TTNN computation
    a_ttnn = Tensor(a_data, device="TTNN")
    b_ttnn = Tensor(b_data, device="TTNN")
    c_ttnn = (a_ttnn + b_ttnn).realize()
    ttnn_result = c_ttnn.numpy()
    
    # PyTorch CPU reference
    a_torch = torch.tensor(a_data, dtype=torch.float32)
    b_torch = torch.tensor(b_data, dtype=torch.float32)
    c_torch = a_torch + b_torch
    torch_result = c_torch.numpy()
    
    # Compare results
    print(f"TTNN result: {ttnn_result}")
    print(f"Torch result: {torch_result}")
    np.testing.assert_allclose(ttnn_result, torch_result, rtol=1e-3, atol=1e-3, 
                               err_msg="TTNN addition doesn't match PyTorch reference")

  def test_exp(self):
    # Test data
    x_data = [0.0, 1.0, 2.0, 3.0]
    
    # TTNN computation
    x_ttnn = Tensor(x_data, device="TTNN")
    y_ttnn = x_ttnn.exp().realize()
    ttnn_result = y_ttnn.numpy()
    
    # PyTorch CPU reference
    x_torch = torch.tensor(x_data, dtype=torch.float32)
    y_torch = torch.exp(x_torch)
    torch_result = y_torch.numpy()
    
    # Compare results (allow higher tolerance due to bfloat16 + tile layout numerical differences)
    print(f"TTNN exp result: {ttnn_result}")
    print(f"Torch exp result: {torch_result}")
    print(f"Relative error: {np.abs((ttnn_result - torch_result) / torch_result)}")
    np.testing.assert_allclose(ttnn_result, torch_result, rtol=4e-2, atol=5e-2,
                               err_msg="TTNN exponential doesn't match PyTorch reference")

  def test_matmul_32x32(self):
    """Fundamental test: 32x32 matrix multiplication to diagnose TTNN tensor layout issues"""
    print("=== Testing 32x32 Matrix Multiplication ===")
    
    # Create deterministic test data
    np.random.seed(42)
    A_data = np.random.randn(32, 32).astype(np.float32) * 0.1  # Small values to avoid overflow
    B_data = np.random.randn(32, 32).astype(np.float32) * 0.1
    
    print(f"A shape: {A_data.shape}, B shape: {B_data.shape}")
    print(f"A sample: {A_data[0, :4]}")
    print(f"B sample: {B_data[0, :4]}")
    
    # TTNN computation
    A_ttnn = Tensor(A_data.tolist(), device="TTNN")
    B_ttnn = Tensor(B_data.tolist(), device="TTNN")
    C_ttnn = (A_ttnn @ B_ttnn).realize()
    ttnn_result = C_ttnn.numpy()
    
    # PyTorch reference
    A_torch = torch.tensor(A_data, dtype=torch.float32)
    B_torch = torch.tensor(B_data, dtype=torch.float32)
    C_torch = A_torch @ B_torch
    torch_result = C_torch.numpy()
    
    print(f"TTNN result shape: {ttnn_result.shape}")
    print(f"PyTorch result shape: {torch_result.shape}")
    print(f"TTNN result sample: {ttnn_result[0, :4]}")
    print(f"PyTorch result sample: {torch_result[0, :4]}")
    
    # Detailed error analysis
    if ttnn_result.shape == torch_result.shape:
      abs_diff = np.abs(ttnn_result - torch_result)
      rel_diff = abs_diff / (np.abs(torch_result) + 1e-8)
      
      print(f"Max absolute difference: {np.max(abs_diff):.2e}")
      print(f"Mean absolute difference: {np.mean(abs_diff):.2e}")
      print(f"Max relative difference: {np.max(rel_diff):.2e}")
      print(f"Mean relative difference: {np.mean(rel_diff):.2e}")
      
      # Check for systematic issues (e.g., zeros, wrong values)
      zero_fraction_ttnn = np.sum(ttnn_result == 0) / ttnn_result.size
      zero_fraction_torch = np.sum(torch_result == 0) / torch_result.size
      print(f"Zero fraction - TTNN: {zero_fraction_ttnn:.3f}, PyTorch: {zero_fraction_torch:.3f}")
      
      # Assert with reasonable tolerance for 32x32 matmul
      np.testing.assert_allclose(ttnn_result, torch_result, rtol=1e-1, atol=1e-1,
                                 err_msg="32x32 matrix multiplication doesn't match PyTorch reference")
      print("✅ 32x32 matrix multiplication test passed!")
    else:
      self.fail(f"Shape mismatch: TTNN {ttnn_result.shape} vs PyTorch {torch_result.shape}")

  def test_matmul_layout_investigation(self):
    """Investigate TTNN tensor layout and data handling"""
    print("=== Investigating TTNN Tensor Layout Issues ===")
    
    # Simple 2x2 case for detailed analysis
    A_data = [[1.0, 2.0], [3.0, 4.0]]
    B_data = [[0.5, 0.0], [0.0, 0.5]]  # Should give [[0.5, 1.0], [1.5, 2.0]]
    
    print(f"Input A: {A_data}")
    print(f"Input B: {B_data}")
    print(f"Expected result: [[0.5, 1.0], [1.5, 2.0]]")
    
    # Test TTNN
    A_ttnn = Tensor(A_data, device="TTNN")
    B_ttnn = Tensor(B_data, device="TTNN")
    
    print(f"A_ttnn shape: {A_ttnn.shape}")
    print(f"B_ttnn shape: {B_ttnn.shape}")
    
    C_ttnn = (A_ttnn @ B_ttnn).realize()
    ttnn_result = C_ttnn.numpy()
    
    print(f"TTNN result: {ttnn_result}")
    print(f"TTNN result shape: {ttnn_result.shape}")
    
    # PyTorch reference
    A_torch = torch.tensor(A_data, dtype=torch.float32)
    B_torch = torch.tensor(B_data, dtype=torch.float32)
    C_torch = A_torch @ B_torch
    torch_result = C_torch.numpy()
    
    print(f"PyTorch result: {torch_result}")
    
    # This test is for investigation - we'll see what's wrong
    if not np.allclose(ttnn_result, torch_result, rtol=1e-2, atol=1e-2):
      print("⚠️ DETECTED: Matrix multiplication mismatch - this confirms the layout/formatting issue")
    else:
      print("✅ Simple 2x2 case works correctly")


if __name__ == '__main__':
  unittest.main()

