import os
import unittest
import numpy as np

from tinygrad import Tensor, Device, nn


@unittest.skipUnless(os.environ.get("TTNN"), "TTNN device not enabled/available")
class TestMNISTTTNN(unittest.TestCase):
  def setUp(self):
    Device.DEFAULT = "TTNN"

  def test_mnist_mlp_forward(self):
    # Simple MNIST-like MLP: flatten 28x28 -> 784, Linear 784->128 -> relu -> Linear 128->10
    class MLP:
      def __init__(self):
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
      def __call__(self, x:Tensor) -> Tensor:
        return self.l2(self.l1(x.flatten(1)).relu())

    model = MLP()
    # random input (batch=1)
    x = Tensor.rand(1,1,28,28, device="TTNN")
    out = model(x).realize()
    arr = out.numpy()
    self.assertEqual(arr.shape, (1,10))


if __name__ == '__main__':
  unittest.main()

