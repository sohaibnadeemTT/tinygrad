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


if __name__ == '__main__':
  unittest.main()

