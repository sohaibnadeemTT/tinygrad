import os
import unittest
import numpy as np
from tinygrad import Tensor, Device, nn
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import torch_load, load_state_dict
from huggingface_hub import hf_hub_download, login
@unittest.skipUnless(os.environ.get("TTNN"), "TTNN device not enabled/available")
class TestMNISTTTNN(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load MNIST test set using tinygrad's built-in loader
    _, _, X_test, Y_test = mnist(device="TTNN")
    cls.X_test = X_test
    cls.Y_test = Y_test
    # Define your model (adjust as needed)
    class MLP:
      def __init__(self):
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 10)
      def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x.flatten(1)).relu()
        return self.l2(x)
    cls.model = MLP()
    # Download weights from Hugging Face using token (if needed)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
      login(token=hf_token)
    # Download the pytorch_model.bin file
    weights_path = hf_hub_download(repo_id="dacorvo/mnist-mlp", filename="pytorch_model.bin", token=hf_token)
    weights = torch_load(weights_path)
    print("Loaded weight keys:", list(weights.keys()))
    key_map = {
      "input_layer.weight": "l1.weight",
      "input_layer.bias": "l1.bias",
      "output_layer.weight": "l2.weight",
      "output_layer.bias": "l2.bias",
    }
    remapped = {key_map.get(k, k): v for k, v in weights.items() if key_map.get(k, k) in ["l1.weight", "l1.bias", "l2.weight", "l2.bias"]}
    load_state_dict(cls.model, remapped)
  def setUp(self):
    Device.DEFAULT = "TTNN"
  def test_mnist_mlp_inference(self):
    # Use the loaded model and test set for inference
    X = self.X_test[:1000]
    Y = self.Y_test[:1000]
    correct = 0
    for i in range(X.shape[0]):
      x = X[i:i+1]
      out = self.model(x).realize()
      pred = int(np.argmax(out.numpy()))
      if pred == int(Y[i].numpy()):
        correct += 1
    accuracy = correct / X.shape[0]
    print(f"MNIST TTNN inference accuracy on {X.shape[0]} samples: {accuracy:.2f}")
    self.assertGreater(accuracy, 0.95)
if __name__ == '__main__':
  unittest.main()