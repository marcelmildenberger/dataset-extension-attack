"""
Parallel processing utilities for DEA performance optimization
"""

import concurrent.futures
import os
import pickle
import psutil
import numpy as np
import torch
from utils import load_dataframe


def load_data_parallel(data_directory, alice_enc_hash, identifier, load_test=False, max_workers=4):
    """
    Load multiple datasets in parallel using ThreadPoolExecutor
    
    Args:
        data_directory: Base data directory
        alice_enc_hash: Alice encoding hash
        identifier: Data identifier
        load_test: Whether to load test data
        max_workers: Maximum number of worker threads
    
    Returns:
        Dictionary containing loaded datasets
    """
    def get_cache_path(data_directory, identifier, alice_enc_hash, name="dataset"):
        os.makedirs(f"{data_directory}/cache", exist_ok=True)
        return os.path.join(data_directory, "cache", f"{name}_{identifier}_{alice_enc_hash}.pkl")

    cache_path = get_cache_path(data_directory, identifier, alice_enc_hash)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data_train, data_val, data_test = pickle.load(f)
        return data_train, data_val, data_test

    # Define file paths
    files_to_load = {
        'reidentified': f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5"
    }
    
    if load_test:
        files_to_load.update({
            'not_reidentified': f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5",
            'all_data': f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5"
        })

    # Load files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(load_dataframe, filepath): key 
            for key, filepath in files_to_load.items()
        }
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                print(f'Loading {key} generated an exception: {exc}')
                results[key] = None

    return results


def log_resource_usage(use_gpu=False):
    """
    Log current CPU, memory, and GPU resource usage
    
    Args:
        use_gpu: Whether to include GPU monitoring
    """
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    memory = psutil.virtual_memory()
    
    print(f"CPU Usage: {np.mean(cpu_percent):.1f}% (avg), cores: {[f'{x:.1f}%' for x in cpu_percent[:8]]}")
    print(f"Memory: {memory.percent:.1f}% ({memory.used//1024//1024}MB used / {memory.total//1024//1024}MB total)")
    
    # GPU usage if available
    if use_gpu and torch.cuda.is_available():
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU Usage: {gpu.load*100:.1f}% | Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        except ImportError:
            # Fallback to torch GPU monitoring
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_cached = torch.cuda.memory_reserved(0)
            print(f"GPU Memory: {gpu_allocated//1024//1024}MB allocated, {gpu_cached//1024//1024}MB cached / {gpu_memory//1024//1024}MB total")


def optimize_torch_performance(use_gpu=False, workers=None):
    """
    Apply PyTorch performance optimizations
    
    Args:
        use_gpu: Whether GPU is available
        workers: Number of CPU workers available
    """
    # Set optimal number of threads for PyTorch
    if workers and workers > 0:
        torch.set_num_threads(min(workers, 16))  # Cap at 16 for optimal performance
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = use_gpu  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    
    if use_gpu and torch.cuda.is_available():
        # Enable Tensor Core usage for mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_optimal_batch_size(dataset_size, gpu_available=False, base_batch_size=32):
    """
    Calculate optimal batch size based on dataset size and hardware
    
    Args:
        dataset_size: Size of the dataset
        gpu_available: Whether GPU is available
        base_batch_size: Base batch size to start from
    
    Returns:
        Optimal batch size
    """
    if gpu_available:
        # Larger batch sizes for GPU
        if dataset_size > 10000:
            return min(128, base_batch_size * 4)
        elif dataset_size > 5000:
            return min(64, base_batch_size * 2)
        else:
            return base_batch_size
    else:
        # Smaller batch sizes for CPU
        if dataset_size > 10000:
            return min(64, base_batch_size * 2)
        else:
            return base_batch_size


def parallel_similarity_computation(data_chunks, metric_func, workers=4):
    """
    Compute similarity matrices in parallel
    
    Args:
        data_chunks: List of data chunks
        metric_func: Similarity metric function
        workers: Number of parallel workers
    
    Returns:
        Combined similarity matrix
    """
    def compute_chunk_similarity(chunk):
        """Compute similarity for a single chunk"""
        return metric_func(chunk)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(compute_chunk_similarity, chunk) for chunk in data_chunks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return np.concatenate(results, axis=0)


def smart_memory_management():
    """
    Implement smart memory management strategies
    """
    # Clear Python garbage collection
    import gc
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PerformanceMonitor:
    """Context manager for monitoring performance during operations"""
    
    def __init__(self, operation_name, use_gpu=False, log_interval=10):
        self.operation_name = operation_name
        self.use_gpu = use_gpu
        self.log_interval = log_interval
        self.start_time = None
        
    def __enter__(self):
        print(f"🚀 Starting {self.operation_name}")
        self.start_time = __import__('time').time()
        log_resource_usage(self.use_gpu)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = __import__('time').time() - self.start_time
        print(f"✅ Completed {self.operation_name} in {elapsed:.2f}s")
        log_resource_usage(self.use_gpu)
        smart_memory_management()


# Export optimization configuration
OPTIMIZATION_CONFIG = {
    "high_performance": {
        "dataloader_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "max_concurrent_trials": 4,
        "batch_size_multiplier": 2
    },
    "memory_optimized": {
        "dataloader_workers": 4,
        "pin_memory": False,
        "persistent_workers": False,
        "max_concurrent_trials": 2,
        "batch_size_multiplier": 1
    },
    "balanced": {
        "dataloader_workers": 6,
        "pin_memory": True,
        "persistent_workers": True,
        "max_concurrent_trials": 3,
        "batch_size_multiplier": 1.5
    }
}