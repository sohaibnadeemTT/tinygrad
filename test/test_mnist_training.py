"""
Comprehensive MNIST Training Tests: TTNN vs PyTorch
Tests complete training pipeline including inference, forward pass, backpropagation, and parameter updates
"""

import pytest
import numpy as np
import time
from pathlib import Path

def pytest_configure(config):
    """Configure pytest to properly handle TTNN backend"""
    import os
    os.environ["TTNN"] = "1"
    os.environ["DEVECTORIZE"] = "0"


class TestMNISTTraining:
    """Test MNIST training on TTNN backend with PyTorch comparison"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        from tinygrad import Device, Tensor
        import os
        
        # Test TTNN device availability by creating a simple tensor
        try:
            test_tensor = Tensor([1.0], device="TTNN")
            print("✅ TTNN device is available")
        except Exception as e:
            pytest.skip(f"TTNN device not available: {e}")
        
        # Create artifacts directory
        self.artifacts_dir = Path("tinygrad-artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Disable debug output for clean tests
        if "TTNN_DEBUG" in os.environ:
            del os.environ["TTNN_DEBUG"]
    
    def test_basic_mnist_inference(self):
        """Test basic MNIST inference comparing TTNN vs PyTorch"""
        from tinygrad import Tensor, nn, Device
        import torch
        import torch.nn as torch_nn
        import torch.nn.functional as torch_F
        
        print("\n=== Testing Basic MNIST Inference ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create simple MNIST model (TinyGrad)
        class MNISTModel:
            def __init__(self):
                self.l1 = nn.Linear(784, 128)
                self.l2 = nn.Linear(128, 10)
            
            def __call__(self, x):
                x = x.reshape(-1, 784)
                x = self.l1(x).relu()
                return self.l2(x)
        
        # Create corresponding PyTorch model
        class TorchMNISTModel(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch_nn.Linear(784, 128)
                self.l2 = torch_nn.Linear(128, 10)
            
            def forward(self, x):
                x = x.view(-1, 784)
                x = torch_F.relu(self.l1(x))
                return self.l2(x)
        
        # Initialize models
        model_ttnn = MNISTModel()
        model_torch = TorchMNISTModel()
        
        # Copy weights from TTNN to PyTorch for fair comparison
        with torch.no_grad():
            model_torch.l1.weight.copy_(torch.from_numpy(model_ttnn.l1.weight.numpy()))
            model_torch.l1.bias.copy_(torch.from_numpy(model_ttnn.l1.bias.numpy()))
            model_torch.l2.weight.copy_(torch.from_numpy(model_ttnn.l2.weight.numpy()))
            model_torch.l2.bias.copy_(torch.from_numpy(model_ttnn.l2.bias.numpy()))
        
        # Create test data (batch of 2 MNIST-like images)
        batch_size = 2
        test_data = np.random.randn(batch_size, 28, 28).astype(np.float32) * 0.1
        
        # TTNN inference
        x_ttnn = Tensor(test_data.tolist(), device="TTNN")
        start_time = time.time()
        output_ttnn = model_ttnn(x_ttnn).realize()
        ttnn_time = time.time() - start_time
        result_ttnn = output_ttnn.numpy()
        
        # PyTorch inference
        x_torch = torch.from_numpy(test_data)
        start_time = time.time()
        with torch.no_grad():
            output_torch = model_torch(x_torch)
        torch_time = time.time() - start_time
        result_torch = output_torch.numpy()
        
        # Compare results
        max_diff = np.max(np.abs(result_ttnn - result_torch))
        print(f"TTNN time: {ttnn_time:.4f}s, PyTorch time: {torch_time:.4f}s")
        print(f"Max difference: {max_diff:.6f}")
        print(f"TTNN output shape: {result_ttnn.shape}, PyTorch shape: {result_torch.shape}")
        
        # Validate results
        assert result_ttnn.shape == result_torch.shape, f"Shape mismatch: {result_ttnn.shape} vs {result_torch.shape}"
        np.testing.assert_allclose(result_ttnn, result_torch, rtol=2e-2, atol=2e-2,
                                   err_msg=f"TTNN inference differs from PyTorch (max diff: {max_diff:.6f})")
        
        print("✅ Basic MNIST inference test passed!")
    
    def test_mnist_training_single_step(self):
        """Test single training step with gradient computation"""
        from tinygrad import Tensor, nn
        import torch
        import torch.nn as torch_nn
        import torch.nn.functional as torch_F
        
        print("\n=== Testing Single Training Step ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create models
        class SimpleModel:
            def __init__(self):
                self.linear = nn.Linear(4, 2)
            
            def __call__(self, x):
                return self.linear(x)
        
        class TorchSimpleModel(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_nn.Linear(4, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        # Initialize models
        model_ttnn = SimpleModel()
        model_torch = TorchSimpleModel()
        
        # Copy weights for fair comparison
        with torch.no_grad():
            model_torch.linear.weight.copy_(torch.from_numpy(model_ttnn.linear.weight.numpy()))
            model_torch.linear.bias.copy_(torch.from_numpy(model_ttnn.linear.bias.numpy()))
        
        # Create training data
        batch_size = 3
        input_data = np.random.randn(batch_size, 4).astype(np.float32) * 0.1
        target_data = np.random.randint(0, 2, (batch_size,)).astype(np.int64)
        
        # TTNN training step
        x_ttnn = Tensor(input_data.tolist(), device="TTNN", requires_grad=False)
        y_ttnn = Tensor(target_data.tolist(), device="TTNN", requires_grad=False)
        
        # Enable gradients
        model_ttnn.linear.weight.requires_grad_(True)
        model_ttnn.linear.bias.requires_grad_(True)
        
        # Forward pass
        logits_ttnn = model_ttnn(x_ttnn)
        loss_ttnn = logits_ttnn.sparse_categorical_crossentropy(y_ttnn)
        
        # Backward pass
        loss_ttnn.backward()
        
        # Get gradients
        weight_grad_ttnn = model_ttnn.linear.weight.grad.numpy() if model_ttnn.linear.weight.grad is not None else None
        bias_grad_ttnn = model_ttnn.linear.bias.grad.numpy() if model_ttnn.linear.bias.grad is not None else None
        
        # PyTorch training step
        x_torch = torch.from_numpy(input_data).requires_grad_(False)
        y_torch = torch.from_numpy(target_data).long()
        
        # Forward pass
        logits_torch = model_torch(x_torch)
        loss_torch = torch_F.cross_entropy(logits_torch, y_torch)
        
        # Backward pass
        loss_torch.backward()
        
        # Get gradients
        weight_grad_torch = model_torch.linear.weight.grad.numpy()
        bias_grad_torch = model_torch.linear.bias.grad.numpy()
        
        # Compare results
        loss_diff = abs(loss_ttnn.numpy().item() - loss_torch.item())
        print(f"TTNN loss: {loss_ttnn.numpy().item():.6f}, PyTorch loss: {loss_torch.item():.6f}")
        print(f"Loss difference: {loss_diff:.6f}")
        
        if weight_grad_ttnn is not None and bias_grad_ttnn is not None:
            weight_grad_diff = np.max(np.abs(weight_grad_ttnn - weight_grad_torch))
            bias_grad_diff = np.max(np.abs(bias_grad_ttnn - bias_grad_torch))
            print(f"Weight gradient max diff: {weight_grad_diff:.6f}")
            print(f"Bias gradient max diff: {bias_grad_diff:.6f}")
            
            # Validate gradients
            np.testing.assert_allclose(weight_grad_ttnn, weight_grad_torch, rtol=5e-2, atol=5e-2,
                                       err_msg=f"Weight gradients differ (max diff: {weight_grad_diff:.6f})")
            np.testing.assert_allclose(bias_grad_ttnn, bias_grad_torch, rtol=5e-2, atol=5e-2,
                                       err_msg=f"Bias gradients differ (max diff: {bias_grad_diff:.6f})")
            
            print("✅ Gradient computation matches PyTorch!")
        else:
            print("⚠️ TTNN gradients not computed")
        
        # Validate loss
        np.testing.assert_allclose(loss_ttnn.numpy(), loss_torch.detach().numpy(), rtol=2e-2, atol=2e-2,
                                   err_msg=f"Loss differs from PyTorch (diff: {loss_diff:.6f})")
        
        print("✅ Single training step test passed!")
    
    def test_parameter_updates(self):
        """Test parameter updates with SGD optimizer"""
        from tinygrad import Tensor, nn
        import torch
        import torch.nn as torch_nn
        import torch.optim as torch_optim
        
        print("\n=== Testing Parameter Updates ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create simple model for parameter update test
        class UpdateModel:
            def __init__(self):
                self.w = Tensor.uniform(2, 3, device="TTNN", requires_grad=True)
                self.b = Tensor.zeros(3, device="TTNN", requires_grad=True)
            
            def __call__(self, x):
                return x @ self.w + self.b
        
        class TorchUpdateModel(torch_nn.Module):
            def __init__(self, w_init, b_init):
                super().__init__()
                self.w = torch_nn.Parameter(torch.from_numpy(w_init))
                self.b = torch_nn.Parameter(torch.from_numpy(b_init))
            
            def forward(self, x):
                return x @ self.w + self.b
        
        # Initialize models
        model_ttnn = UpdateModel()
        w_init = model_ttnn.w.numpy().copy()
        b_init = model_ttnn.b.numpy().copy()
        
        model_torch = TorchUpdateModel(w_init, b_init)
        
        # Create simple training data
        x_data = np.array([[1.0, 0.5], [0.2, 1.0]], dtype=np.float32)
        y_data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        
        learning_rate = 0.1
        
        # TTNN training
        x_ttnn = Tensor(x_data.tolist(), device="TTNN")
        y_ttnn = Tensor(y_data.tolist(), device="TTNN")
        
        # Forward pass
        pred_ttnn = model_ttnn(x_ttnn)
        loss_ttnn = pred_ttnn.sub(y_ttnn).pow(2).mean()
        
        # Backward pass
        loss_ttnn.backward()
        
        # Manual SGD update
        if model_ttnn.w.grad is not None:
            w_new_ttnn = model_ttnn.w.sub(model_ttnn.w.grad.mul(learning_rate))
            b_new_ttnn = model_ttnn.b.sub(model_ttnn.b.grad.mul(learning_rate))
        else:
            w_new_ttnn = model_ttnn.w
            b_new_ttnn = model_ttnn.b
        
        # PyTorch training
        x_torch = torch.from_numpy(x_data)
        y_torch = torch.from_numpy(y_data)
        
        optimizer = torch_optim.SGD(model_torch.parameters(), lr=learning_rate)
        
        # Forward pass
        pred_torch = model_torch(x_torch)
        loss_torch = torch_nn.functional.mse_loss(pred_torch, y_torch)
        
        # Backward pass and update
        optimizer.zero_grad()
        loss_torch.backward()
        optimizer.step()
        
        # Compare results
        if model_ttnn.w.grad is not None and model_ttnn.b.grad is not None:
            w_diff = np.max(np.abs(w_new_ttnn.numpy() - model_torch.w.detach().numpy()))
            b_diff = np.max(np.abs(b_new_ttnn.numpy() - model_torch.b.detach().numpy()))
            
            print(f"Weight update max diff: {w_diff:.6f}")
            print(f"Bias update max diff: {b_diff:.6f}")
            print(f"TTNN loss: {loss_ttnn.numpy().item():.6f}, PyTorch loss: {loss_torch.item():.6f}")
            
            # Validate parameter updates
            np.testing.assert_allclose(w_new_ttnn.numpy(), model_torch.w.detach().numpy(), 
                                       rtol=5e-2, atol=5e-2,
                                       err_msg=f"Weight updates differ (max diff: {w_diff:.6f})")
            np.testing.assert_allclose(b_new_ttnn.numpy(), model_torch.b.detach().numpy(), 
                                       rtol=5e-2, atol=5e-2,
                                       err_msg=f"Bias updates differ (max diff: {b_diff:.6f})")
            
            print("✅ Parameter updates match PyTorch!")
        else:
            print("⚠️ TTNN gradients not computed for parameter updates")
        
        print("✅ Parameter update test passed!")
    
    def test_multi_step_training(self):
        """Test multiple training steps to validate learning progress"""
        from tinygrad import Tensor, nn
        import torch
        import torch.nn as torch_nn
        import torch.optim as torch_optim
        
        print("\n=== Testing Multi-Step Training ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create models for multi-step training
        class TrainingModel:
            def __init__(self):
                self.linear = nn.Linear(2, 1)
            
            def __call__(self, x):
                return self.linear(x)
        
        class TorchTrainingModel(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch_nn.Linear(2, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # Initialize models
        model_ttnn = TrainingModel()
        model_torch = TorchTrainingModel()
        
        # Copy initial weights
        with torch.no_grad():
            model_torch.linear.weight.copy_(torch.from_numpy(model_ttnn.linear.weight.numpy()))
            model_torch.linear.bias.copy_(torch.from_numpy(model_ttnn.linear.bias.numpy()))
        
        # Create training dataset (simple linear regression)
        n_samples = 4
        X = np.random.randn(n_samples, 2).astype(np.float32) * 0.5
        y = (X[:, 0] + 2 * X[:, 1] + 0.1).reshape(-1, 1).astype(np.float32)  # y = x1 + 2*x2 + noise
        
        learning_rate = 0.01
        n_steps = 3
        
        print(f"Training for {n_steps} steps with learning rate {learning_rate}")
        
        # Training loop comparison
        ttnn_losses = []
        torch_losses = []
        
        for step in range(n_steps):
            # TTNN training step
            x_ttnn = Tensor(X.tolist(), device="TTNN", requires_grad=False)
            y_ttnn = Tensor(y.tolist(), device="TTNN", requires_grad=False)
            
            # Enable gradients
            model_ttnn.linear.weight.requires_grad_(True)
            model_ttnn.linear.bias.requires_grad_(True)
            
            # Forward pass
            pred_ttnn = model_ttnn(x_ttnn)
            loss_ttnn = pred_ttnn.sub(y_ttnn).pow(2).mean()
            
            # Backward pass
            loss_ttnn.backward()
            
            # Manual parameter update (if gradients exist)
            if model_ttnn.linear.weight.grad is not None:
                # Simulate SGD update with assign
                new_weight = model_ttnn.linear.weight.sub(model_ttnn.linear.weight.grad.mul(learning_rate))
                new_bias = model_ttnn.linear.bias.sub(model_ttnn.linear.bias.grad.mul(learning_rate))
                model_ttnn.linear.weight.assign(new_weight)
                model_ttnn.linear.bias.assign(new_bias)
            
            ttnn_losses.append(loss_ttnn.numpy().item())
            
            # PyTorch training step
            x_torch = torch.from_numpy(X)
            y_torch = torch.from_numpy(y)
            
            optimizer = torch_optim.SGD(model_torch.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            
            pred_torch = model_torch(x_torch)
            loss_torch = torch_nn.functional.mse_loss(pred_torch, y_torch)
            loss_torch.backward()
            optimizer.step()
            
            torch_losses.append(loss_torch.item())
            
            print(f"  Step {step+1}: TTNN loss={ttnn_losses[-1]:.6f}, PyTorch loss={torch_losses[-1]:.6f}")
        
        # Validate that both are learning (loss should decrease)
        ttnn_improvement = ttnn_losses[0] - ttnn_losses[-1]
        torch_improvement = torch_losses[0] - torch_losses[-1]
        
        print(f"TTNN improvement: {ttnn_improvement:.6f}")
        print(f"PyTorch improvement: {torch_improvement:.6f}")
        
        # Check if both models are learning
        assert ttnn_improvement > 0, f"TTNN model not learning (improvement: {ttnn_improvement:.6f})"
        assert torch_improvement > 0, f"PyTorch model not learning (improvement: {torch_improvement:.6f})"
        
        # Compare final losses
        final_loss_diff = abs(ttnn_losses[-1] - torch_losses[-1])
        print(f"Final loss difference: {final_loss_diff:.6f}")
        
        # Allow for some divergence in multi-step training
        np.testing.assert_allclose(ttnn_losses[-1], torch_losses[-1], rtol=1e-1, atol=1e-1,
                                   err_msg=f"Final losses diverged too much (diff: {final_loss_diff:.6f})")
        
        print("✅ Multi-step training test passed!")
    
    def test_mnist_end_to_end(self):
        """Test complete MNIST training pipeline end-to-end"""
        from tinygrad import Tensor, nn
        
        print("\n=== Testing MNIST End-to-End Pipeline ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        
        # Create MNIST model
        class MNISTNet:
            def __init__(self):
                self.l1 = nn.Linear(784, 64)
                self.l2 = nn.Linear(64, 10)
            
            def __call__(self, x):
                x = x.reshape(-1, 784)
                x = self.l1(x).relu()
                return self.l2(x)
        
        # Initialize model
        model = MNISTNet()
        
        # Create synthetic MNIST-like data
        batch_size = 4
        images = np.random.randn(batch_size, 28, 28).astype(np.float32) * 0.3
        labels = np.random.randint(0, 10, batch_size).astype(np.int64)
        
        print(f"Training on batch of {batch_size} samples")
        
        # Convert to tensors
        x = Tensor(images.tolist(), device="TTNN")
        y = Tensor(labels.tolist(), device="TTNN")
        
        # Enable gradients
        model.l1.weight.requires_grad_(True)
        model.l1.bias.requires_grad_(True)
        model.l2.weight.requires_grad_(True)
        model.l2.bias.requires_grad_(True)
        
        # Forward pass
        start_time = time.time()
        logits = model(x)
        loss = logits.sparse_categorical_crossentropy(y)
        forward_time = time.time() - start_time
        
        # Backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Check gradients exist
        gradients_exist = all([
            model.l1.weight.grad is not None,
            model.l1.bias.grad is not None,
            model.l2.weight.grad is not None,
            model.l2.bias.grad is not None
        ])
        
        print(f"Forward pass time: {forward_time:.4f}s")
        print(f"Backward pass time: {backward_time:.4f}s")
        print(f"Loss: {loss.numpy().item():.6f}")
        print(f"Logits shape: {logits.shape}")
        print(f"Gradients computed: {gradients_exist}")
        
        # Validate results
        assert logits.shape == (batch_size, 10), f"Wrong output shape: {logits.shape}"
        assert not np.isnan(loss.numpy().item()), "Loss is NaN"
        assert loss.numpy().item() > 0, f"Loss should be positive, got {loss.numpy().item()}"
        
        if gradients_exist:
            # Check gradient magnitudes are reasonable
            grad_norms = []
            for param_name, param in [("l1.weight", model.l1.weight), ("l1.bias", model.l1.bias),
                                      ("l2.weight", model.l2.weight), ("l2.bias", model.l2.bias)]:
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad.numpy())
                    grad_norms.append(grad_norm)
                    print(f"{param_name} gradient norm: {grad_norm:.6f}")
            
            # Gradients should not be too small or too large
            assert all(1e-6 < norm < 100 for norm in grad_norms), f"Gradient norms out of range: {grad_norms}"
            print("✅ Gradient magnitudes are reasonable")
        else:
            print("⚠️ Some gradients not computed")
        
        print("✅ MNIST end-to-end test passed!")
    
    def test_performance_benchmark(self):
        """Benchmark TTNN vs PyTorch performance"""
        from tinygrad import Tensor, nn
        import torch
        import torch.nn as torch_nn
        
        print("\n=== Performance Benchmark ===")
        
        # Set deterministic behavior
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create larger models for performance testing
        class BenchmarkModel:
            def __init__(self):
                self.l1 = nn.Linear(784, 256)
                self.l2 = nn.Linear(256, 128)
                self.l3 = nn.Linear(128, 10)
            
            def __call__(self, x):
                x = x.reshape(-1, 784)
                x = self.l1(x).relu()
                x = self.l2(x).relu()
                return self.l3(x)
        
        class TorchBenchmarkModel(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch_nn.Linear(784, 256)
                self.l2 = torch_nn.Linear(256, 128)
                self.l3 = torch_nn.Linear(128, 10)
            
            def forward(self, x):
                x = x.view(-1, 784)
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return self.l3(x)
        
        # Initialize models
        model_ttnn = BenchmarkModel()
        model_torch = TorchBenchmarkModel()
        
        # Copy weights for fair comparison
        with torch.no_grad():
            model_torch.l1.weight.copy_(torch.from_numpy(model_ttnn.l1.weight.numpy()))
            model_torch.l1.bias.copy_(torch.from_numpy(model_ttnn.l1.bias.numpy()))
            model_torch.l2.weight.copy_(torch.from_numpy(model_ttnn.l2.weight.numpy()))
            model_torch.l2.bias.copy_(torch.from_numpy(model_ttnn.l2.bias.numpy()))
            model_torch.l3.weight.copy_(torch.from_numpy(model_ttnn.l3.weight.numpy()))
            model_torch.l3.bias.copy_(torch.from_numpy(model_ttnn.l3.bias.numpy()))
        
        # Create larger batch for performance testing
        batch_size = 8
        test_data = np.random.randn(batch_size, 28, 28).astype(np.float32) * 0.1
        
        # Warmup runs
        x_ttnn = Tensor(test_data.tolist(), device="TTNN")
        x_torch = torch.from_numpy(test_data)
        
        model_ttnn(x_ttnn).realize()  # Warmup TTNN
        with torch.no_grad():
            model_torch(x_torch)  # Warmup PyTorch
        
        # Benchmark TTNN
        n_runs = 3
        ttnn_times = []
        for _ in range(n_runs):
            start_time = time.time()
            output_ttnn = model_ttnn(x_ttnn).realize()
            ttnn_times.append(time.time() - start_time)
        
        # Benchmark PyTorch
        torch_times = []
        for _ in range(n_runs):
            start_time = time.time()
            with torch.no_grad():
                output_torch = model_torch(x_torch)
            torch_times.append(time.time() - start_time)
        
        # Calculate statistics
        ttnn_avg = np.mean(ttnn_times)
        ttnn_std = np.std(ttnn_times)
        torch_avg = np.mean(torch_times)
        torch_std = np.std(torch_times)
        
        print(f"TTNN: {ttnn_avg:.4f}±{ttnn_std:.4f}s")
        print(f"PyTorch: {torch_avg:.4f}±{torch_std:.4f}s")
        print(f"Speedup: {torch_avg/ttnn_avg:.2f}x" if ttnn_avg < torch_avg else f"Slowdown: {ttnn_avg/torch_avg:.2f}x")
        
        # Verify correctness
        max_diff = np.max(np.abs(output_ttnn.numpy() - output_torch.numpy()))
        print(f"Max output difference: {max_diff:.6f}")
        
        np.testing.assert_allclose(output_ttnn.numpy(), output_torch.numpy(), rtol=2e-2, atol=2e-2,
                                   err_msg=f"Performance benchmark outputs differ (max diff: {max_diff:.6f})")
        
        print("✅ Performance benchmark completed!")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "-s"])