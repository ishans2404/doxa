"""Device manager for automatic CPU/GPU selection."""

import numpy as np

class DeviceManager:
    """Manages automatic device selection and operations."""

    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        # minimum elements for GPU acceleration
        self.memory_threshold = 1000
        self.operation_preferences = {
            'matmul' : 'gpu',
            'conv2d': 'gpu',
            'neural_network' : 'gpu',
            'small_ops' : 'cpu'
        }

    def _check_gpu_availability(self) -> bool:
        """Check if GPU (CUDA) is available."""
        try:
            import cupy as cp
            # test basic GPU operation
            _ = cp.cuda.Device(0).compute_capability
            return True
        except (ImportError, Exception):
            return False
    
    def select_optimal_device(
            self,
            tensor_size: int,
            operation_type: str = 'general'
    ) -> str:
        """Automatically select the best device based on data and operation."""
        # If GPU not available, always use CPU
        if not self.gpu_available:
            return 'cpu'
        
        # Small tensors should use CPU
        if tensor_size < self.memory_threshold:
            return 'cpu'
        
        # Check operation preferences
        preferred_device = self.operation_preferences.get(operation_type, 'cpu')

        # For large tensors and Gpu-preferred operations, use GPU
        if tensor_size > self.memory_threshold and preferred_device == 'gpu':
            return 'gpu'
        
        # Default to CPU
        return 'cpu'
    
    def get_backend_module(self, device: str):
        """Get the appropriate backend module (numpy or cupy)."""
        if device == 'gpu' and self.gpu_available:
            try:
                import cupy as cp
                return cp
            except ImportError:
                return np
        return np
    
    def transfer_to_device(self, array, transfer_device: str):
        """Transfer array to the specified device."""
        if transfer_device == 'gpu' and self.gpu_available:
            # Transfer to GPU
            try:
                import cupy as cp
                if isinstance(array, np.ndarray):
                    return cp.asarray(array)
                return array
            except ImportError:
                return array
        else:
            # Transfer to CPU
            if hasattr(array, 'get'):  # cupy array
                return array.get()
            return np.asarray(array)
    
    def get_device_info(self) -> dict:
        """Get information about available devices."""
        info = {
            'gpu_available' : self.gpu_available,
            'memory_threshold' : self.memory_threshold,
        }

        if self.gpu_available:
            try:
                import cupy as cp
                info['gpu_name'] = cp.cuda.Device().name.decode()
                info['gpu_memory'] = cp.cuda.Device().mem_info
                info['gpu_compute_capability'] = cp.cuda.Device().compute_capability
                info['num_gpus'] = cp.cuda.runtime.getDeviceCount()
                info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            except:
                info['gpu_info'] = 'Could not retrieve GPU details.'
        
        return info
        


