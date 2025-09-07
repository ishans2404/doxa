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
from .core.tensor import Tensor
from .core.device_manager import DeviceManager
from .core import utils
from .utils import metrics

# Key Classes available at the package level
__all__ = [
    "Tensor",
    "DeviceManager",
    "utils",
    "__version__",
    "metrics",
]

# Initialize global device manager
_device_manager = DeviceManager()

def get_device_manager():
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        from .backend.device_manager import DeviceManager
        _device_manager = DeviceManager()
    return _device_manager

# Convenience functions
def tensor(data, dtype=None, device='auto'):
    """Create a doxa tensor with automatic device selection."""
    return Tensor(data, dtype=dtype, device=device)

def cuda_available():
    """Check if CUDA is available."""
    return get_device_manager().gpu_available()
