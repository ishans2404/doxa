"""Global pytest configuration and fixtures."""

import pytest
import numpy as np
from doxa import Tensor
from doxa import DeviceManager

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    np.random.seed(331)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 1.0, -0.5]) + 0.1 * np.random.randn(100)
    return X, y

@pytest.fixture
def small_data():
    """Provide small dataset for quick tests."""
    np.random.seed(331)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    return X, y

@pytest.fixture
def device_manager():
    """Fixture to provide a DeviceManager instance."""
    return DeviceManager()

@pytest.fixture
def tensor_2d():
    """Fixture to provide a 2D Tensor."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    return Tensor(data)

@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except:
        return False
    
def pytest_configuration(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: tests requiring GPU")
    config.addinivalue_line("markers", "slow: slow running tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: units tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["gpu", "slow", "integration"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
