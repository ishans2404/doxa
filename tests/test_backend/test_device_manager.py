import pytest
import numpy as np
from doxa.backend.device_manager import DeviceManager


class TestDeviceManager:
    @pytest.fixture
    def device_manager(self):
        return DeviceManager()

    def test_gpu_availability(self, device_manager):
        """Test that GPU is available and CUDA works."""
        assert device_manager.gpu_available is True
        
        # Verify CuPy can be imported
        import cupy as cp
        assert cp is not None
        
        # Test basic CUDA operation
        gpu_array = cp.array([1, 2, 3])
        result = gpu_array * 2
        assert cp.array_equal(result, cp.array([2, 4, 6]))

    def test_device_selection(self, device_manager):
        """Test device selection logic for different scenarios."""
        # Test small tensor (should use CPU)
        small_size = 500
        assert device_manager.select_optimal_device(small_size) == 'cpu'
        
        # Test large tensor with GPU operation (should use GPU)
        large_size = 2000
        assert device_manager.select_optimal_device(large_size, 'matmul') == 'gpu'
        
        # Test large tensor with CPU operation
        assert device_manager.select_optimal_device(large_size, 'small_ops') == 'cpu'

    def test_backend_module_selection(self, device_manager):
        """Test backend module selection (numpy vs cupy)."""
        import cupy as cp
        
        # Test GPU backend
        gpu_backend = device_manager.get_backend_module('gpu')
        assert gpu_backend == cp
        
        # Test CPU backend
        cpu_backend = device_manager.get_backend_module('cpu')
        assert cpu_backend == np

    def test_array_transfer(self, device_manager):
        """Test array transfer between CPU and GPU."""
        import cupy as cp
        
        # Create test array on CPU
        cpu_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        # Transfer to GPU
        gpu_array = device_manager.transfer_to_device(cpu_array, 'gpu')
        assert isinstance(gpu_array, cp.ndarray)
        assert gpu_array.device.id == 0  # Check if on GPU
        
        # Verify data integrity
        assert np.array_equal(cpu_array, cp.asnumpy(gpu_array))
        
        # Transfer back to CPU
        cpu_array_back = device_manager.transfer_to_device(gpu_array, 'cpu')
        assert isinstance(cpu_array_back, np.ndarray)
        assert np.array_equal(cpu_array, cpu_array_back)

    def test_gpu_operations(self, device_manager):
        """Test actual GPU computations."""
        import cupy as cp
        
        # Create large matrices to test GPU performance
        size = 2000
        cpu_matrix = np.random.rand(size, size).astype(np.float32)
        
        # Transfer to GPU
        gpu_matrix = device_manager.transfer_to_device(cpu_matrix, 'gpu')
        
        # Perform matrix multiplication on GPU
        result_gpu = cp.dot(gpu_matrix, gpu_matrix)
        
        # Transfer back to CPU
        result_cpu = device_manager.transfer_to_device(result_gpu, 'cpu')
        
        assert isinstance(result_cpu, np.ndarray)
        assert result_cpu.shape == (size, size)

    def test_device_info(self, device_manager):
        """Test GPU device information retrieval."""
        info = device_manager.get_device_info()
        
        # Verify all GPU information is available
        assert info['gpu_available'] is True
        assert isinstance(info['gpu_name'], str)
        assert len(info['gpu_name']) > 0
        assert isinstance(info['gpu_memory'], tuple)
        assert isinstance(info['gpu_compute_capability'], tuple)
        assert isinstance(info['num_gpus'], int)
        assert info['num_gpus'] > 0
        assert isinstance(info['cuda_version'], int)
        
        # Print GPU info for debugging
        print("\nGPU Information:")
        for key, value in info.items():
            print(f"{key}: {value}")

    def test_memory_operations(self, device_manager):
        """Test memory-intensive operations."""
        import cupy as cp
        
        # Create large array
        large_array = np.random.rand(5000, 5000).astype(np.float32)
        
        # Transfer to GPU
        gpu_array = device_manager.transfer_to_device(large_array, 'gpu')
        
        # Perform operations
        gpu_result = cp.sum(gpu_array, axis=1)
        cpu_result = device_manager.transfer_to_device(gpu_result, 'cpu')
        
        assert isinstance(cpu_result, np.ndarray)
        assert len(cpu_result.shape) == 1
        assert cpu_result.shape[0] == 5000

    def test_operation_preferences(self, device_manager):
        """Test operation preferences configuration."""
        assert device_manager.operation_preferences['matmul'] == 'gpu'
        assert device_manager.operation_preferences['conv2d'] == 'gpu'
        assert device_manager.operation_preferences['neural_network'] == 'gpu'
        assert device_manager.operation_preferences['small_ops'] == 'cpu'
        
        # Test device selection based on operation type
        large_size = 2000
        for op_type, preferred_device in device_manager.operation_preferences.items():
            selected_device = device_manager.select_optimal_device(large_size, op_type)
            assert selected_device == preferred_device
