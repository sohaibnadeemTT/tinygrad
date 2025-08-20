import os
import unittest
import numpy as np

from tinygrad import Tensor, Device


@unittest.skipUnless(os.environ.get("TTNN"), "TTNN device not enabled/available")
class TestTTNNBackend(unittest.TestCase):
  def setUp(self):
    # ensure default device is TTNN for these tests
    Device.DEFAULT = "TTNN"

  def test_add(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0], device="TTNN")
    b = Tensor([10.0, 20.0, 30.0, 40.0], device="TTNN")
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32), rtol=1e-3, atol=1e-3)

  def test_exp(self):
    x = Tensor([0.0, 1.0, 2.0, 3.0], device="TTNN")
    y = x.exp().realize()
    # allow higher tolerance due to bfloat16 + tile layout numerical differences
    np.testing.assert_allclose(y.numpy(), np.exp(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)), rtol=4e-2, atol=5e-2)


if __name__ == '__main__':
  unittest.main()

