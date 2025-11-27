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
            'gpu_available': False,
            'memory_threshold': self.memory_threshold,
            'gpu_info': None,
            'gpu_name': 'No GPU available',
            'gpu_memory': (0, 0),
            'gpu_compute_capability': (0, 0),
            'num_gpus': 0,
            'cuda_version': 0
        }
        
        if self.gpu_available:
            try:
                import cupy as cp
                
                # Get device count first
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count == 0:
                    info['gpu_info'] = 'No CUDA devices found'
                    return info
                
                # Use device 0 (primary GPU)
                with cp.cuda.Device(0):
                    # Correct API calls
                    info['gpu_available'] = True
                    info['num_gpus'] = device_count
                    info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
                    
                    # Get device properties
                    properties = cp.cuda.runtime.getDeviceProperties(0)
                    info['gpu_name'] = properties['name'].decode('utf-8')
                    info['gpu_compute_capability'] = (properties['major'], properties['minor'])
                    
                    # Get memory info
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    info['gpu_memory'] = (free_mem, total_mem)
                    
                    info['gpu_info'] = None
                    
            except Exception as e:
                info['gpu_info'] = f'Could not retrieve GPU details: {str(e)}'
        
        return info

        


