"""Tests for Tensor class."""

import pytest
import numpy as np
from doxa import Tensor

class TestTensor:
    """Test cases for Tensor class."""

    def test_tensor_creation(self, tensor_2d):
        """Test basic tensor creation."""
        assert tensor_2d.shape == (2, 3)
        np.testing.assert_array_equal(
            tensor_2d.numpy(),
            np.array([[1, 2, 3], [4, 5, 6]])
        )

    def test_tensor_device_selection(self):
        """Test automatic device selection."""
        # Small tensor should use CPU
        small_tensor = Tensor(np.random.randn(10, 10))
        assert small_tensor.device == 'cpu'

    def test_tensor_operations(self):
        """Test tensor mathematical operations."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 0], [1, 2]])

        # Addition
        c = a + b
        expected = np.array([[3, 2], [4, 6]])
        np.testing.assert_array_equal(c.numpy(), expected)

        # Multiplication
        d = a * b
        expected = np.array([[2, 0], [3, 8]])
        np.testing.assert_array_equal(d.numpy(), expected)
    
    def test_tensor_matmul(self):
        """Test matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 0], [1, 2]])

        c = a @ b
        expected = np.array([[4, 4], [10, 8]])
        np.testing.assert_array_equal(c.numpy(), expected)

    @pytest.mark.gpu
    def test_gpu_operation(self, gpu_available):
        """Test GPU operations if available."""
        if not gpu_available:
            pytest.skip("GPU not available.")
        
        tensor = Tensor([[1, 2], [3, 4]])
        gpu_tensor = tensor.to_gpu()
        assert gpu_tensor.device == 'gpu'

        # Test moving back to CPU
        cpu_tensor =  gpu_tensor.to_cpu()
        assert cpu_tensor.device == 'cpu'
        np.testing.assert_array_equal(
            cpu_tensor.numpy(),
            tensor.numpy()
        )

    def test_tensor_dtype_preservation(self):
        """Test that data types are preserved."""
        data = np.array([[1.5, 2.5]], dtype=np.float32)
        tensor = Tensor(data)
        assert tensor.dtype == np.float32
    
    @pytest.mark.parametrize("shape", [
        (10,), (10, 5), (10, 5, 3), (2, 2, 2, 2)
    ])
    def test_tensor_shapes(self, shape):
        """Test tensor with various shapes."""
        data = np.random.randn(*shape)
        tensor = Tensor(data)
        assert tensor.shape == shape
