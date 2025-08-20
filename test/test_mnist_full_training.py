"""
COMPREHENSIVE MNIST TRAINING ON TTNN BACKEND
=============================================

This test demonstrates a complete MNIST training pipeline using the TTNN backend:
1. Load real MNIST data from MNIST_ORG folder
2. Create a neural network model
3. Train for multiple epochs with loss tracking
4. Validate model performance
5. Demonstrate inference on sample inputs

All operations run on the TTNN device with no fallbacks.
"""

import numpy as np
import struct
import gzip
import os
from pathlib import Path
import time
import pytest


def pytest_configure(config):
    """Configure pytest for TTNN backend"""
    import os
    os.environ["TTNN"] = "1"
    os.environ["DEVECTORIZE"] = "0"


class MNISTLoader:
    """MNIST data loader for the MNIST_ORG folder format"""
    
    def __init__(self, data_dir="MNIST_ORG"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MNIST data directory {data_dir} not found")
    
    def _read_idx_file(self, filename):
        """Read IDX format files (MNIST standard format)"""
        filepath = self.data_dir / filename
        
        # Try both compressed and uncompressed versions
        if filepath.with_suffix('.gz').exists():
            with gzip.open(filepath.with_suffix('.gz'), 'rb') as f:
                data = f.read()
        elif filepath.exists():
            with open(filepath, 'rb') as f:
                data = f.read()
        else:
            raise FileNotFoundError(f"Could not find {filepath} or {filepath}.gz")
        
        # Parse IDX header
        magic = struct.unpack('>I', data[:4])[0]
        ndim = magic & 0xFF
        dtype_map = {0x08: 'uint8', 0x09: 'int8', 0x0B: 'int16', 0x0C: 'int32', 0x0D: 'float32', 0x0E: 'float64'}
        dtype = dtype_map.get((magic >> 8) & 0xFF, 'uint8')
        
        # Read dimensions
        dims = struct.unpack('>' + 'I' * ndim, data[4:4 + 4 * ndim])
        
        # Read data
        offset = 4 + 4 * ndim
        array = np.frombuffer(data[offset:], dtype=np.dtype(dtype).newbyteorder('>'))
        return array.reshape(dims)
    
    def load_train_data(self, max_samples=None):
        """Load training images and labels"""
        images = self._read_idx_file('train-images.idx3-ubyte')
        labels = self._read_idx_file('train-labels.idx1-ubyte')
        
        if max_samples:
            images = images[:max_samples]
            labels = labels[:max_samples]
        
        # Normalize images to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        return images, labels
    
    def load_test_data(self, max_samples=None):
        """Load test images and labels"""
        images = self._read_idx_file('t10k-images.idx3-ubyte')
        labels = self._read_idx_file('t10k-labels.idx1-ubyte')
        
        if max_samples:
            images = images[:max_samples]
            labels = labels[:max_samples]
        
        # Normalize images to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        return images, labels


class MNISTModel:
    """Simple but effective MNIST model for TTNN backend"""
    
    def __init__(self, device="TTNN"):
        from tinygrad import Tensor
        
        self.device = device
        
        # Network architecture: 784 -> 128 -> 64 -> 10
        # Initialize with Xavier/Glorot initialization for better training
        self.w1 = Tensor.uniform(784, 128, device=device, requires_grad=True) * (2.0 / (784 + 128)) ** 0.5
        self.b1 = Tensor.zeros(128, device=device, requires_grad=True)
        
        self.w2 = Tensor.uniform(128, 64, device=device, requires_grad=True) * (2.0 / (128 + 64)) ** 0.5
        self.b2 = Tensor.zeros(64, device=device, requires_grad=True)
        
        self.w3 = Tensor.uniform(64, 10, device=device, requires_grad=True) * (2.0 / (64 + 10)) ** 0.5
        self.b3 = Tensor.zeros(10, device=device, requires_grad=True)
    
    def __call__(self, x):
        """Forward pass"""
        # Flatten input: (batch_size, 28, 28) -> (batch_size, 784)
        x = x.reshape(-1, 784)
        
        # Layer 1: 784 -> 128 with ReLU
        x = (x @ self.w1 + self.b1).relu()
        
        # Layer 2: 128 -> 64 with ReLU  
        x = (x @ self.w2 + self.b2).relu()
        
        # Layer 3: 64 -> 10 (logits)
        x = x @ self.w3 + self.b3
        
        return x
    
    def parameters(self):
        """Return all trainable parameters"""
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
    
    def zero_grad(self):
        """Zero out gradients"""
        for param in self.parameters():
            param.grad = None


class SGDOptimizer:
    """Simple SGD optimizer for TTNN backend"""
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Update parameters using gradients"""
        for param in self.parameters:
            if param.grad is not None:
                # param = param - lr * grad
                param.assign(param - param.grad * self.lr)
    
    def zero_grad(self):
        """Zero out gradients"""
        for param in self.parameters:
            param.grad = None


class TestMNISTFullTraining:
    """Complete MNIST training test on TTNN backend"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        from tinygrad import Tensor
        
        # Test TTNN device availability
        try:
            test_tensor = Tensor([1.0], device="TTNN")
            print("✅ TTNN device is available")
        except Exception as e:
            pytest.skip(f"TTNN device not available: {e}")
        
        # Setup data loader
        try:
            self.mnist_loader = MNISTLoader()
            print("✅ MNIST data loader initialized")
        except Exception as e:
            pytest.skip(f"MNIST data not available: {e}")
    
    def test_mnist_data_loading(self):
        """Test MNIST data loading functionality"""
        print("\n=== Testing MNIST Data Loading ===")
        
        # Load small sample for testing
        train_images, train_labels = self.mnist_loader.load_train_data(max_samples=100)
        test_images, test_labels = self.mnist_loader.load_test_data(max_samples=20)
        
        print(f"Train data shape: {train_images.shape}, labels: {train_labels.shape}")
        print(f"Test data shape: {test_images.shape}, labels: {test_labels.shape}")
        print(f"Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
        print(f"Label range: [{train_labels.min()}, {train_labels.max()}]")
        
        # Verify data integrity
        assert train_images.shape == (100, 28, 28), f"Wrong train image shape: {train_images.shape}"
        assert train_labels.shape == (100,), f"Wrong train label shape: {train_labels.shape}"
        assert test_images.shape == (20, 28, 28), f"Wrong test image shape: {test_images.shape}"
        assert test_labels.shape == (20,), f"Wrong test label shape: {test_labels.shape}"
        assert 0 <= train_images.min() and train_images.max() <= 1, "Images not normalized"
        assert 0 <= train_labels.min() and train_labels.max() <= 9, "Invalid label range"
        
        print("✅ MNIST data loading test passed!")
    
    def test_model_initialization(self):
        """Test model initialization and forward pass"""
        print("\n=== Testing Model Initialization ===")
        
        from tinygrad import Tensor
        
        # Create model
        model = MNISTModel(device="TTNN")
        
        # Test forward pass with sample data
        batch_size = 4
        sample_input = Tensor.uniform(batch_size, 28, 28, device="TTNN")
        
        # Forward pass
        output = model(sample_input)
        result = output.realize().numpy()
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Verify output shape and properties
        assert result.shape == (batch_size, 10), f"Wrong output shape: {result.shape}"
        assert not np.isnan(result).any(), "NaN in model output"
        assert not np.isinf(result).any(), "Inf in model output"
        
        # Test parameter access
        params = model.parameters()
        print(f"Model has {len(params)} parameters")
        for i, param in enumerate(params):
            print(f"  Param {i}: shape {param.shape}, device {param.device}")
        
        print("✅ Model initialization test passed!")
    
    def test_training_single_epoch(self):
        """Test single epoch training"""
        print("\n=== Testing Single Epoch Training ===")
        
        from tinygrad import Tensor
        
        # Load small dataset for quick testing
        train_images, train_labels = self.mnist_loader.load_train_data(max_samples=50)
        
        # Create model and optimizer
        model = MNISTModel(device="TTNN")
        optimizer = SGDOptimizer(model.parameters(), lr=0.1)
        
        # Training parameters
        batch_size = 10
        num_batches = len(train_images) // batch_size
        
        print(f"Training on {len(train_images)} samples, {num_batches} batches")
        
        total_loss = 0.0
        correct_predictions = 0
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_images = train_images[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]
            
            # Convert to tensors
            x = Tensor(batch_images.tolist(), device="TTNN")
            y = Tensor(batch_labels.astype(np.int32).tolist(), device="TTNN")
            
            # Forward pass
            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            loss_val = loss.realize().numpy().item()
            total_loss += loss_val
            
            # Calculate accuracy
            predictions = logits.realize().numpy().argmax(axis=1)
            correct_predictions += (predictions == batch_labels).sum()
            
            print(f"  Batch {batch_idx + 1}/{num_batches}: Loss = {loss_val:.4f}")
        
        # Final metrics
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / len(train_images)
        
        print(f"Epoch completed: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")
        
        # Verify training is working
        assert not np.isnan(avg_loss), "Loss is NaN"
        assert avg_loss > 0, "Loss should be positive"
        assert 0 <= accuracy <= 1, f"Invalid accuracy: {accuracy}"
        
        print("✅ Single epoch training test passed!")
    
    def test_full_mnist_training(self):
        """Complete MNIST training with multiple epochs"""
        print("\n" + "="*60)
        print("🚀 STARTING FULL MNIST TRAINING ON TTNN BACKEND")
        print("="*60)
        
        from tinygrad import Tensor
        
        # Load training data (use subset for reasonable test time)
        print("📂 Loading MNIST dataset...")
        train_images, train_labels = self.mnist_loader.load_train_data(max_samples=1000)
        test_images, test_labels = self.mnist_loader.load_test_data(max_samples=200)
        
        print(f"   Training samples: {len(train_images)}")
        print(f"   Test samples: {len(test_images)}")
        
        # Create model and optimizer
        print("🧠 Initializing model...")
        model = MNISTModel(device="TTNN")
        optimizer = SGDOptimizer(model.parameters(), lr=0.01)  # Lower learning rate for stability
        
        # Training configuration
        num_epochs = 3
        batch_size = 32
        num_batches = len(train_images) // batch_size
        
        print(f"📋 Training configuration:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches per epoch: {num_batches}")
        print(f"   Learning rate: {optimizer.lr}")
        
        # Training history
        train_losses = []
        train_accuracies = []
        
        print("\n🏋️ Starting training...")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx in range(num_batches):
                batch_start_time = time.time()
                
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_images = train_images[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                
                # Convert to TTNN tensors
                x = Tensor(batch_images.tolist(), device="TTNN")
                y = Tensor(batch_labels.astype(np.int32).tolist(), device="TTNN")
                
                # Forward pass
                logits = model(x)
                loss = logits.sparse_categorical_crossentropy(y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                loss_val = loss.realize().numpy().item()
                predictions = logits.realize().numpy().argmax(axis=1)
                batch_correct = (predictions == batch_labels).sum()
                
                # Update epoch metrics
                epoch_loss += loss_val
                epoch_correct += batch_correct
                epoch_total += len(batch_labels)
                
                batch_time = time.time() - batch_start_time
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                    batch_acc = batch_correct / len(batch_labels)
                    print(f"   Batch {batch_idx + 1:2d}/{num_batches}: "
                          f"Loss = {loss_val:.4f}, "
                          f"Acc = {batch_acc:.3f}, "
                          f"Time = {batch_time:.2f}s")
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches
            epoch_accuracy = epoch_correct / epoch_total
            epoch_time = time.time() - epoch_start_time
            
            train_losses.append(avg_loss)
            train_accuracies.append(epoch_accuracy)
            
            print(f"\n✅ Epoch {epoch + 1} completed:")
            print(f"   Average Loss: {avg_loss:.4f}")
            print(f"   Training Accuracy: {epoch_accuracy:.3f} ({epoch_correct}/{epoch_total})")
            print(f"   Time: {epoch_time:.1f}s")
        
        print("\n🎯 Training completed! Final results:")
        print(f"   Final Loss: {train_losses[-1]:.4f}")
        print(f"   Final Training Accuracy: {train_accuracies[-1]:.3f}")
        print(f"   Loss improvement: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        print(f"   Accuracy improvement: {train_accuracies[0]:.3f} -> {train_accuracies[-1]:.3f}")
        
        # Validation on test set
        print("\n🧪 Evaluating on test set...")
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        # Evaluate in smaller batches to avoid memory issues
        eval_batch_size = 20
        num_eval_batches = len(test_images) // eval_batch_size
        
        for batch_idx in range(num_eval_batches):
            start_idx = batch_idx * eval_batch_size
            end_idx = start_idx + eval_batch_size
            
            batch_images = test_images[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]
            
            # Convert to tensors
            x = Tensor(batch_images.tolist(), device="TTNN")
            y = Tensor(batch_labels.astype(np.int32).tolist(), device="TTNN")
            
            # Forward pass (no gradients needed)
            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)
            
            # Calculate metrics
            predictions = logits.realize().numpy().argmax(axis=1)
            test_correct += (predictions == batch_labels).sum()
            test_total += len(batch_labels)
            test_loss += loss.realize().numpy().item()
        
        test_accuracy = test_correct / test_total
        avg_test_loss = test_loss / num_eval_batches
        
        print(f"📊 Test Results:")
        print(f"   Test Loss: {avg_test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.3f} ({test_correct}/{test_total})")
        
        # Demonstrate inference on individual samples
        print("\n🔍 Demonstrating inference on sample inputs...")
        self._demonstrate_inference(model, test_images[:5], test_labels[:5])
        
        # Verify training worked
        assert train_accuracies[-1] > train_accuracies[0], "Model should improve during training"
        assert train_losses[-1] < train_losses[0], "Loss should decrease during training"
        assert test_accuracy > 0.5, f"Test accuracy too low: {test_accuracy:.3f}"
        
        print("\n🎉 MNIST TRAINING COMPLETED SUCCESSFULLY!")
        print("✅ All assertions passed - model trained effectively on TTNN backend")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'model': model
        }
    
    def _demonstrate_inference(self, model, images, labels):
        """Demonstrate inference on sample inputs"""
        from tinygrad import Tensor
        
        print("\n🎭 Inference demonstration:")
        print("-" * 30)
        
        for i in range(len(images)):
            # Single image inference
            image = images[i:i+1]  # Keep batch dimension
            true_label = labels[i]
            
            # Convert to tensor
            x = Tensor(image.tolist(), device="TTNN")
            
            # Forward pass
            logits = model(x)
            probabilities = logits.realize().numpy()[0]  # Remove batch dimension
            
            # Get prediction
            predicted_label = probabilities.argmax()
            confidence = probabilities.max()
            
            # Show results
            status = "✅" if predicted_label == true_label else "❌"
            print(f"   Sample {i+1}: True={true_label}, Pred={predicted_label}, "
                  f"Conf={confidence:.3f} {status}")
        
        print()


if __name__ == "__main__":
    # Run the full training test directly
    import sys
    
    # Setup environment
    os.environ["TTNN"] = "1" 
    os.environ["DEVECTORIZE"] = "0"
    
    print("🚀 Running MNIST Full Training Test")
    print("=" * 50)
    
    try:
        # Setup manually
        from tinygrad import Tensor
        
        # Test TTNN device availability
        test_tensor = Tensor([1.0], device="TTNN")
        print("✅ TTNN device is available")
        
        # Setup data loader
        mnist_loader = MNISTLoader()
        print("✅ MNIST data loader initialized")
        
        # Create test instance and run training
        test_instance = TestMNISTFullTraining()
        test_instance.mnist_loader = mnist_loader
        
        # Run the full training
        results = test_instance.test_full_mnist_training()
        
        print("\n🎊 SUCCESS: Full MNIST training completed!")
        print(f"Final test accuracy: {results['test_accuracy']:.3f}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)