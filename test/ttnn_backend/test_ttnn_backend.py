import os
import unittest
import numpy as np
from tinygrad import Tensor, Device
import math
@unittest.skipUnless(os.environ.get("TTNN"), "TTNN device not enabled/available")
class TestTTNNBackend(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    print("\n" + "🚀"*30)
    print("🚀         TTNN BACKEND TEST SUITE         🚀")
    print("🚀    Testing vectorized operations with   🚀")
    print("🚀          DEVECTORIZE=0 enabled          🚀")
    print("🚀"*30)
    
  def setUp(self):
    # ensure default device is TTNN for these tests
    Device.DEFAULT = "TTNN"
  def test_add(self):
    print("\n" + "="*60)
    print("🧪 TEST: Addition (4D vectors)")
    print("="*60)
    print("Testing: [1,2,3,4] + [10,20,30,40] = [11,22,33,44]")
    
    a = Tensor([1.0, 2.0, 3.0, 4.0], device="TTNN")
    b = Tensor([10.0, 20.0, 30.0, 40.0], device="TTNN")
    c = (a + b).realize()
    result = c.numpy()
    expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("✅ ADD test completed")
    
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
  def test_exp(self):
    print("\n" + "="*60)
    print("🧪 TEST: Exponential function")
    print("="*60)
    print("Testing: exp([0,1,2,3]) = [1, e, e², e³]")
    
    x = Tensor([0.0, 1.0, 2.0, 3.0], device="TTNN")
    y = x.exp().realize()
    result = y.numpy()
    expected = np.exp(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("⚠️  EXP test (may have comparison operation issues)")
    
    # allow higher tolerance due to bfloat16 + tile layout numerical differences
    np.testing.assert_allclose(result, expected, rtol=4e-2, atol=5e-2)
  def test_sub(self):
    print("\n" + "="*60)
    print("🧪 TEST: Subtraction (4D vectors)")
    print("="*60)
    print("Testing: [5,7,9,10] - [2,3,4,3] = [3,4,5,7]")
    
    a = Tensor([5.0, 7.0, 9.0, 10.0], device="TTNN")
    b = Tensor([2.0, 3.0, 4.0, 3.0], device="TTNN")
    c = (a - b).realize()
    result = c.numpy()
    expected = np.array([3.0, 4.0, 5.0, 7.0], dtype=np.float32)
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("✅ SUB test completed")
    
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
  def test_mul_4d(self):
    print("\n" + "="*60)
    print("🧪 TEST: Multiplication (4D vectors)")
    print("="*60)
    print("Testing: [2,3,4,5] * [5,6,7,8] = [10,18,28,40]")
    
    a = Tensor([2.0, 3.0, 4.0, 5.0], device="TTNN")
    b = Tensor([5.0, 6.0, 7.0, 8.0], device="TTNN")
    c = (a * b).realize()
    result = c.numpy()
    expected = np.array([10.0, 18.0, 28.0, 40.0], dtype=np.float32)
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("✅ MUL_4D test completed")
    
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
  def test_mul_3d(self):
    print("\n" + "="*60)
    print("🧪 TEST: Multiplication (3D vectors)")
    print("="*60)
    print("Testing: [2,3,4] * [2,3,4] = [4,9,16]")
    
    a = Tensor([2.0, 3.0, 4.0], device="TTNN")
    b = Tensor([2.0, 3.0, 4.0], device="TTNN")
    c = (a * b).realize()
    result = c.numpy()
    expected = np.array([4.0, 9.0, 16.0], dtype=np.float32)
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("✅ MUL_3D test completed")
    
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
  def test_mul_2d(self):
    print("\n" + "="*60)
    print("🧪 TEST: Multiplication (2D vectors)")
    print("="*60)
    print("Testing: [2,3] * [5,6] = [10,18]")
    
    a = Tensor([2.0, 3.0], device="TTNN")
    b = Tensor([5.0, 6.0], device="TTNN")
    c = (a * b).realize()
    result = c.numpy()
    expected = np.array([10.0, 18.0], dtype=np.float32)
    
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print("✅ MUL_2D test completed")
    
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
if __name__ == '__main__':
  print("\n" + "🎯"*30)
  print("🎯       STARTING TTNN BACKEND TESTS       🎯")
  print("🎯"*30)
  
  unittest.main()
  
  print("\n" + "🏁"*30)
  print("🏁        TTNN BACKEND TESTS COMPLETE       🏁")
  print("🏁"*30)