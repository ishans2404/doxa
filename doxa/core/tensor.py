"""Tensor implementation with automatic device management."""

import numpy as np
from typing import Any, Optional, Union, Tuple
from ..backend.device_manager import DeviceManager

class Tensor:
    """
    Doxa Tensor with automatic device management.
    - Automatic CPU/GPU selection.
    - Seamless device transfers.
    - NumPy-compatible operations.
    """

    def __init__(
            self,
            data,
            dtype: Optional[Any] = None,
            device: str = 'auto'
    ):
        """
        Initialize a Doxa Tensor.

        Args:
            data: Input data (list, np.ndarray, etc.)
            dtype: Desired data type (e.g., np.float32)
            device: Device placement ('auto', 'cpu', 'gpu')
        """
        # Convert to numpy array first
        self._data = np.asarray(data, dtype=dtype)

        # Initialize device manager
        self._device_manager = DeviceManager()

        # Select and set device
        self._device = self._select_device(device)

        # Move data to selected device 
        self._move_to_device(self._device)

        # Gradient support 
        self._grad = None
        self._requires_grad = False

    def _select_device(self, device: str) -> str:
        """Select the optimal device based on policy."""
        if device == 'auto':
            return self._device_manager.select_optimal_device(
                self._data.size,
                'general'
            )
        elif device in ['cpu', 'gpu']:
            if device == 'gpu' and not self._device_manager.gpu_available:
                print("Warning: GPU not available. Falling back to CPU.")
                return 'cpu'
            return device
        else:
            raise ValueError(f"Invalid device: {device}. Use 'auto', 'cpu', or 'gpu'")
    
    def _move_to_device(self, device: str):
        """Move the tensor data to the specified device."""
        self._data = self._device_manager.transfer_to_device(self._data, device)
    
    @property
    def data(self):
        """Get the underlying data array."""
        return self._data
    
    @property
    def device(self) -> str:
        """Get the current device of the tensor."""
        return self._device
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return self._data.shape
    
    @property
    def dtype(self):
        """Get the data type of the tensor."""
        return self._data.dtype
    
    @property
    def requires_grad(self) -> bool:
        """Check if gradient tracking is enabled."""
        return self._requires_grad
    
    def to_cpu(self) -> 'Tensor':
        """Transfer tensor to CPU."""
        if self._device != 'cpu':
            cpu_data = self._device_manager.transfer_to_device(self._data, 'cpu')
            result = Tensor.__new__(Tensor)
            result._data = cpu_data
            result._device = 'cpu'
            result._device_manager = self._device_manager
            result._grad = self._grad
            result._requires_grad = self._requires_grad
            return result
        return self
    
    def to_gpu(self) -> 'Tensor':
        """Transfer tensor to GPU."""
        if not self._device_manager.gpu_available:
            raise RuntimeError("GPU not available.")
        if self._device != 'gpu':
            gpu_data = self._device_manager.transfer_to_device(self._data, 'gpu')
            result = Tensor.__new__(Tensor)
            result._data = gpu_data
            result._device = 'gpu'
            result._device_manager = self._device_manager
            result._grad = self._grad
            result._requires_grad = self._requires_grad
            return result
        return self
    
    # Mathematical operations
    def __add__(self, other):
        """Element wise addition."""
        if isinstance(other, Tensor):
            # Ensure both tensors on same device
            if self._device != other._device:
                other = other.to_cpu() if other._device == 'gpu' else other.to_gpu()
            result_data = self._data + other._data
        else:
            result_data = self._data + other
        
        return Tensor(result_data, device=self._device)
    
    def __mul__(self, other):
        """Element wise multiplication."""
        if isinstance(other, Tensor):
            # Ensure both tensors on same device
            if self._device != other._device:
                other = other.to_cpu() if other._device == 'gpu' else other.to_gpu()
            result_data = self._data * other._data
        else:
            result_data = self._data * other
        
        return Tensor(result_data, device=self._device)
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, Tensor):
            # Ensure both tensors on same device
            if self._device != other._device:
                other = other.to_cpu() if other._device == 'gpu' else other.to_gpu()
            result_data = self._data @ other._data
        else:
            result_data = self._data @ other
        
        return Tensor(result_data, device=self._device)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor(data={self._data}, device={self._device})"
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (moves to CPU if needed)."""
        if self._device == 'gpu' and hasattr(self._data, 'get'):
            return self._data.get()
        return np.asarray(self._data)
    
# Convenience functions
def zeros(shape: Tuple[int, ...], dtype=None, device='auto') -> 'Tensor':
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape=shape, dtype=dtype), device=device)

def ones(shape: Tuple[int, ...], dtype=None, device='auto') -> 'Tensor':
    """Create a tensor filled with ones."""
    return Tensor(np.ones(shape=shape, dtype=dtype), device=device)

def randn(*shape, dtype=None, device='auto') -> 'Tensor':
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype), device=device)
