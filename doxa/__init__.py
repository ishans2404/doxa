"""
Doxa: A modular, intelligent ML/DL library with automatic device management.

Features:
- Intelligent CPU/GPU device selection
- Unified algorithm interfaces
- Plugin-based extensibility
- Memory-efficient processing
"""

__version__ = "0.0.1"
__author__ = "Ishan Singh"

# Core imports
from .core.tensor import Tensor, ones, randn, zeros
from .backend.device_manager import DeviceManager
from .utils import metrics

# Key Classes available at the package level
__all__ = [
    "Tensor", 
    "DeviceManager",
    "metrics",
    "cuda_available", 
    "get_device_manager",
    "__version__",
    'randn',
    'zeros',
    'ones',
]

# Initialize global device manager
_device_manager = None

def get_device_manager():
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager

# Convenience functions
def tensor(data, dtype=None, device='auto'):
    """Create a doxa tensor with automatic device selection."""
    return Tensor(data, dtype=dtype, device=device)

def cuda_available():
    """Check if CUDA is available."""
    return get_device_manager().gpu_available

# Initialize the global device manager on import
_device_manager = DeviceManager()