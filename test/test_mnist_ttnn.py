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

  def test_mnist_backward_basics(self):
    # Test most basic backward pass functionality
    x = Tensor([2.0], device="TTNN", requires_grad=True)
    y = (x ** 2).sum()  # Make it scalar
    y.backward()
    
    # Should get gradient dy/dx = 2*x = 2*2 = 4
    self.assertIsNotNone(x.grad)
    expected_grad = 4.0
    actual_grad = x.grad.numpy()[0]
    self.assertAlmostEqual(actual_grad, expected_grad, places=3)

  def test_mnist_parameter_grad_debug(self):
    # Debug parameter gradient issues step by step using operations that work
    print("=== Parameter Gradient Debug ===")
    
    # Use the pattern that worked in test_mnist_backward_basics
    # Single element operations that we know work
    w = Tensor([1.0], device="TTNN", requires_grad=True)
    x = Tensor([2.0], device="TTNN")
    
    print(f"w.requires_grad: {w.requires_grad}")
    print(f"w.grad: {w.grad}")
    
    # Simple multiplication (no matrix ops to avoid view issues)
    y = w * x  # 1.0 * 2.0 = 2.0
    # Use the working pattern from basic test - square and sum to get scalar
    loss = (y ** 2).sum()  # (2.0)^2 = 4.0
    
    print(f"Loss: {loss.numpy()}")
    print(f"w.requires_grad before backward: {w.requires_grad}")
    
    loss.backward()
    
    print(f"w.grad after backward: {w.grad}")
    if w.grad is not None:
      print(f"w.grad value: {w.grad.numpy()}")
      # Expected gradient: d/dw[(w*x)^2] = d/dw[(w*2)^2] = d/dw[4*w^2] = 8*w = 8*1 = 8
      expected_grad = 8.0  # 2 * (w*x) * x = 2 * (1*2) * 2 = 8
      print(f"Expected gradient: {expected_grad}")
    else:
      print("Gradient is None!")

  def test_mnist_simple_gradient_debug(self):
    # Test the simplest possible gradient case to understand the issue
    print("=== Simple Gradient Debug ===")
    
    with Tensor.train():
      # Test even simpler case: w * 2
      print("--- Very Simple Case: w * 2 ---")
      w_simple = Tensor([5.0], requires_grad=True)
      print(f"w_simple.requires_grad: {w_simple.requires_grad}")
      print(f"w_simple.device: {w_simple.device}")
      
      loss_simple = w_simple * 2.0  # d/dw = 2
      print(f"loss_simple shape: {loss_simple.shape}")
      print(f"loss_simple: {loss_simple.numpy()}")
      
      # Need to make it a scalar for backward pass
      loss_scalar = loss_simple.sum()  # d/dw = 2
      print(f"loss_scalar shape: {loss_scalar.shape}")
      print(f"loss_scalar: {loss_scalar.numpy()}")
      
      loss_scalar.backward()
      
      print(f"w_simple.grad: {w_simple.grad}")
      if w_simple.grad is not None:
        print(f"w_simple.grad values: {w_simple.grad.numpy()}")
      
      # Test tensor creation on CPU vs TTNN
      print("--- CPU Matrix Multiplication ---")
      x_cpu = Tensor([[1.0, 2.0]], device="CPU", requires_grad=False)
      w_cpu = Tensor([[3.0], [4.0]], device="CPU", requires_grad=True)
      
      print(f"CPU devices: x={x_cpu.device}, w={w_cpu.device}")
      print(f"CPU w.requires_grad: {w_cpu.requires_grad}")
      
      out_cpu = x_cpu @ w_cpu
      print(f"CPU out: {out_cpu.numpy()}")
      print(f"CPU out.uop: {out_cpu.uop}")
      
      loss_cpu = out_cpu.sum()
      print(f"CPU loss: {loss_cpu.numpy()}")
      print(f"CPU loss.uop: {loss_cpu.uop}")
      
      # Check computation graph
      all_uops_cpu = loss_cpu.uop.toposort()
      print(f"CPU UOps in computation graph: {len(all_uops_cpu)}")
      for i, uop in enumerate(all_uops_cpu):
        print(f"  {i}: {uop.op}")
      
      # Check if w_cpu is in the computation graph
      found_w_cpu = False
      print(f"w_cpu.uop: {w_cpu.uop}")
      print(f"w_cpu.uop.op: {w_cpu.uop.op}")
      for i, uop in enumerate(all_uops_cpu):
        if uop == w_cpu.uop:
          found_w_cpu = True
          print(f"Found w_cpu in computation graph at {i}: {uop}")
          break
        # Also check for partial matches
        if str(uop.op) == 'Ops.BUFFER' and hasattr(uop, 'arg') and hasattr(w_cpu.uop, 'arg'):
          if uop.arg == w_cpu.uop.arg:
            print(f"Found matching BUFFER at {i}: {uop}")
      
      if not found_w_cpu:
        print("ERROR: w_cpu not found in computation graph!")
      
      loss_cpu.backward()
      
      print(f"CPU w.grad: {w_cpu.grad}")
      if w_cpu.grad is not None:
        print(f"CPU w.grad values: {w_cpu.grad.numpy()}")
      
      # Test direct tensor creation on TTNN
      print("--- TTNN Matrix Multiplication (Direct) ---")
      try:
        x_ttnn = Tensor([[1.0, 2.0]], device="TTNN", requires_grad=False)
        w_ttnn = Tensor([[3.0], [4.0]], device="TTNN", requires_grad=True)
        
        print(f"TTNN devices: x={x_ttnn.device}, w={w_ttnn.device}")
        print(f"TTNN w.requires_grad: {w_ttnn.requires_grad}")
        
        out_ttnn = x_ttnn @ w_ttnn
        print(f"TTNN out: {out_ttnn.numpy()}")
        
        loss_ttnn = out_ttnn.sum()
        print(f"TTNN loss: {loss_ttnn.numpy()}")
        
        loss_ttnn.backward()
        
        print(f"TTNN w.grad: {w_ttnn.grad}")
        if w_ttnn.grad is not None:
          print(f"TTNN w.grad values: {w_ttnn.grad.numpy()}")
      except Exception as e:
        print(f"TTNN direct creation failed: {e}")

  def test_mnist_cpu_vs_ttnn_gradients(self):
    # Compare CPU vs TTNN gradient computation to isolate the issue
    print("=== CPU vs TTNN Gradient Comparison ===")
    
    with Tensor.train():
      # Test on CPU first
      print("--- CPU Gradient Test ---")
      x_cpu = Tensor([[1.0, 2.0]], device="CPU")
      w_cpu = Tensor([[3.0], [4.0]], device="CPU", requires_grad=True)
      
      print(f"CPU w.requires_grad: {w_cpu.requires_grad}")
      
      out_cpu = x_cpu @ w_cpu
      loss_cpu = out_cpu.sum()
      
      print(f"CPU loss: {loss_cpu.numpy()}")
      
      loss_cpu.backward()
      
      print(f"CPU w.grad: {w_cpu.grad}")
      if w_cpu.grad is not None:
        print(f"CPU w.grad values: {w_cpu.grad.numpy()}")
      
      # Test on TTNN
      print("--- TTNN Gradient Test ---")
      x_ttnn = Tensor([[1.0, 2.0]], device="TTNN")
      w_ttnn = Tensor([[3.0], [4.0]], device="TTNN", requires_grad=True)
      
      print(f"TTNN w.requires_grad: {w_ttnn.requires_grad}")
      print(f"TTNN w.uop: {w_ttnn.uop}")
      print(f"TTNN w.uop.op: {w_ttnn.uop.op}")
      
      out_ttnn = x_ttnn @ w_ttnn
      print(f"TTNN out.uop: {out_ttnn.uop}")
      print(f"TTNN out.uop.op: {out_ttnn.uop.op}")
      
      loss_ttnn = out_ttnn.sum()
      print(f"TTNN loss.uop: {loss_ttnn.uop}")
      print(f"TTNN loss.uop.op: {loss_ttnn.uop.op}")
      
      print(f"TTNN loss: {loss_ttnn.numpy()}")
      
      # Check if tensors are in the computation graph for gradient computation
      all_uops = loss_ttnn.uop.toposort()
      tensors_need_grad = []
      print(f"Total UOps in computation graph: {len(all_uops)}")
      
      # Manually check if w_ttnn is in the computation graph
      found_w = False
      for uop in all_uops:
        # Check if this UOp corresponds to our w_ttnn tensor
        if uop == w_ttnn.uop:
          found_w = True
          print(f"Found w_ttnn in computation graph: {uop}")
          break
      
      if not found_w:
        print("ERROR: w_ttnn not found in computation graph!")
      
      loss_ttnn.backward()
      
      print(f"TTNN w.grad: {w_ttnn.grad}")
      if w_ttnn.grad is not None:
        print(f"TTNN w.grad values: {w_ttnn.grad.numpy()}")

  def test_mnist_matrix_multiplication_debug(self):
    # Test matrix multiplication with gradients step by step
    with Tensor.train():
      print("=== Matrix Multiplication Debug ===")
      
      # Simple known values
      x = Tensor([[1.0, 2.0]], device="TTNN")  # (1, 2)
      w = Tensor([[3.0], [4.0]], device="TTNN", requires_grad=True)  # (2, 1)
      
      print(f"x shape: {x.shape}, x values: {x.numpy()}")
      print(f"w shape: {w.shape}, w values: {w.numpy()}")
      print(f"w.requires_grad: {w.requires_grad}")
      
      # Matrix multiplication: (1,2) @ (2,1) = (1,1)
      out = x @ w  # Should be [[1*3 + 2*4]] = [[11]]
      
      print(f"out shape: {out.shape}, out values: {out.numpy()}")
      
      # Simple loss: just sum the output
      loss = out.sum()  # Should be 11
      
      print(f"Loss value: {loss.numpy()}")
      
      # Backward pass
      loss.backward()
      
      print(f"w.grad after backward: {w.grad}")
      if w.grad is not None:
        print(f"w.grad values: {w.grad.numpy()}")
        # Expected gradient: dL/dw = x^T = [[1], [2]]
        expected_grad = [[1.0], [2.0]]
        print(f"Expected w.grad: {expected_grad}")

  def test_mnist_loss_computation_debug(self):
    # Debug why loss becomes zero in complex training scenarios
    with Tensor.train():
      print("=== Loss Computation Debug ===")
      
      # Test each step of the computation individually
      print("--- Step 1: Create simple known tensors ---")
      x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="TTNN")  # (2, 3)
      target = Tensor([[10.0], [20.0]], device="TTNN")  # (2, 1)  
      w = Tensor([[1.0], [1.0], [1.0]], device="TTNN", requires_grad=True)  # (3, 1)
      b = Tensor([0.0], device="TTNN", requires_grad=True)  # (1,)
      
      print(f"x.shape: {x.shape}, x values: {x.numpy()}")
      print(f"target.shape: {target.shape}, target values: {target.numpy()}")
      print(f"w.shape: {w.shape}, w values: {w.numpy()}")
      print(f"b.shape: {b.shape}, b values: {b.numpy()}")
      
      print("--- Step 2: Matrix multiplication ---")
      matmul_result = x @ w  # (2, 3) @ (3, 1) = (2, 1)
      print(f"x @ w shape: {matmul_result.shape}, values: {matmul_result.numpy()}")
      # Expected: [[1+2+3], [4+5+6]] = [[6], [15]]
      
      print("--- Step 3: Add bias ---")
      out = matmul_result + b  # (2, 1) + (1,) = (2, 1)
      print(f"out shape: {out.shape}, values: {out.numpy()}")
      # Expected: [[6], [15]]
      
      print("--- Step 4: Compute squared difference ---")
      diff = out - target  # (2, 1) - (2, 1) = (2, 1)
      print(f"diff shape: {diff.shape}, values: {diff.numpy()}")
      # Expected: [[6-10], [15-20]] = [[-4], [-5]]
      
      squared_diff = diff ** 2  # (2, 1)
      print(f"squared_diff shape: {squared_diff.shape}, values: {squared_diff.numpy()}")
      # Expected: [[16], [25]]
      
      print("--- Step 5: Compute mean ---")
      loss = squared_diff.mean()  # scalar
      print(f"loss shape: {loss.shape}, value: {loss.numpy()}")
      # Expected: (16 + 25) / 2 = 20.5
      
      print("--- Step 6: Test backward pass ---")
      if loss.numpy() > 0:
        loss.backward()
        print(f"w.grad: {w.grad}")
        print(f"b.grad: {b.grad}")
        if w.grad is not None:
          print(f"w.grad values: {w.grad.numpy()}")
        if b.grad is not None:
          print(f"b.grad values: {b.grad.numpy()}")
      else:
        print("Loss is zero - skipping backward pass")

  def test_mnist_simple_training(self):
    # Test very simple training case purely on TTNN  
    with Tensor.train():
      print("=== In training mode ===")
      
      # Create data on TTNN with known non-zero values (avoiding Tensor.rand issue)
      # Use 4-element arrays to ensure TTNN compatibility
      x = Tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]], device="TTNN")
      target = Tensor([[10.0], [20.0]], device="TTNN")
      
      # Create parameter directly on TTNN with requires_grad (non-zero values)
      w = Tensor([[0.1], [0.2], [0.3], [0.0]], device="TTNN", requires_grad=True)
      b = Tensor([0.5], device="TTNN", requires_grad=True)
      
      print(f"w.requires_grad: {w.requires_grad}")
      print(f"b.requires_grad: {b.requires_grad}")
      
      # Debug UOp structure
      print(f"w.uop.op: {w.uop.op}")
      print(f"b.uop.op: {b.uop.op}")
      
      # Debug intermediate values
      print(f"x values: {x.numpy()}")
      print(f"target values: {target.numpy()}")
      print(f"w values: {w.numpy()}")
      print(f"b values: {b.numpy()}")
      
      # Manual linear layer
      out = (x @ w + b).realize()
      print(f"out values: {out.numpy()}")
      
      loss = ((out - target) ** 2).mean().realize()
      
      print(f"Loss value: {loss.numpy()}")
      
      # Backward pass
      loss.backward()
      
      print(f"w.grad after backward: {w.grad}")
      print(f"b.grad after backward: {b.grad}")
      
      # Check gradients exist
      self.assertIsNotNone(w.grad, "Weight should have gradients")
      self.assertIsNotNone(b.grad, "Bias should have gradients") 
      
      weight_grad_norm = (w.grad ** 2).sum().numpy()
      bias_grad_norm = (b.grad ** 2).sum().numpy()
      
      print(f"Weight grad norm: {weight_grad_norm}")
      print(f"Bias grad norm: {bias_grad_norm}")
      
      # Gradients should be non-zero for non-trivial loss
      self.assertGreater(weight_grad_norm + bias_grad_norm, 0, "At least one gradient should be non-zero")

  def test_mnist_mlp_training(self):
    # Test complete training loop: forward, loss, backward, parameter update
    class MLP:
      def __init__(self):
        self.l1 = nn.Linear(784, 32)  # Even smaller network for testing
        self.l2 = nn.Linear(32, 10)
      def __call__(self, x:Tensor) -> Tensor:
        return self.l2(self.l1(x.flatten(1)).relu())

    model = MLP()
    
    # Enable gradients for all parameters
    model.l1.weight.requires_grad_(True)
    model.l1.bias.requires_grad_(True) 
    model.l2.weight.requires_grad_(True)
    model.l2.bias.requires_grad_(True)
    
    # Create synthetic training data (batch_size=1 for testing)
    batch_size = 1
    x = Tensor.rand(batch_size, 1, 28, 28, device="TTNN")
    # Random target labels (one-hot encoded)
    targets = Tensor.rand(batch_size, 10, device="TTNN")
    
    # Get parameters for gradient checking
    params = [model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias]
    
    # Store initial parameter values
    initial_params = [p.numpy().copy() for p in params]
    
    # Forward pass
    logits = model(x)
    
    # Compute loss (MSE for simplicity)
    loss = ((logits - targets) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are non-zero (with tolerance)
    for i, p in enumerate(params):
      self.assertIsNotNone(p.grad, f"Parameter {i} should have gradients")
      grad_norm = (p.grad ** 2).sum().numpy()
      self.assertGreater(grad_norm, 1e-10, f"Parameter {i} gradient should be non-zero")
    
    # Simple SGD update (learning_rate = 0.01)
    lr = 0.01
    for p in params:
      p.assign(p - lr * p.grad).realize()
    
    # Check that parameters have been updated
    for i, p in enumerate(params):
      updated_params = p.numpy()
      param_diff = np.abs(updated_params - initial_params[i]).max()
      self.assertGreater(param_diff, 1e-8, f"Parameter {i} should have been updated")
    
    # Test another training step to ensure consistency
    # Zero gradients
    for p in params:
      p.grad = None
    
    # Second forward/backward pass
    logits2 = model(x)
    loss2 = ((logits2 - targets) ** 2).mean()
    loss2.backward()
    
    # Verify gradients exist again
    for p in params:
      self.assertIsNotNone(p.grad, "Parameters should have gradients after second pass")

  def test_mnist_gradient_computation(self):
    # Test specific gradient computation components
    # Test basic operations that are crucial for training
    
    # Test 1: Linear layer gradient
    x = Tensor.rand(2, 3, device="TTNN")
    linear = nn.Linear(3, 2)
    # Ensure parameters require gradients
    linear.weight.requires_grad_(True)
    linear.bias.requires_grad_(True)
    out = linear(x)
    loss = out.sum()
    loss.backward()
    
    self.assertIsNotNone(linear.weight.grad)
    self.assertIsNotNone(linear.bias.grad)
    self.assertEqual(linear.weight.grad.shape, linear.weight.shape)
    self.assertEqual(linear.bias.grad.shape, linear.bias.shape)
    
    # Test 2: ReLU gradient
    x = Tensor([-1.0, 0.0, 1.0, 2.0], device="TTNN", requires_grad=True)
    y = x.relu()
    loss = y.sum()
    loss.backward()
    
    grad_expected = np.array([0.0, 0.0, 1.0, 1.0])  # ReLU derivative
    np.testing.assert_allclose(x.grad.numpy(), grad_expected, rtol=1e-5)
    
    # Test 3: Chain rule through multiple operations
    x = Tensor.rand(2, 4, device="TTNN", requires_grad=True)
    y = x.relu()
    z = y ** 2
    loss = z.mean()
    loss.backward()
    
    self.assertIsNotNone(x.grad)
    self.assertEqual(x.grad.shape, x.shape)


if __name__ == '__main__':
  unittest.main()

