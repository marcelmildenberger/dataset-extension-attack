# Performance Optimization Analysis for DEA Codebase

## Executive Summary

This analysis identifies key performance bottlenecks in the Data Extraction and Analysis (DEA) codebase and provides specific recommendations for optimizing parallel processing across 19 CPUs and 1 GPU. The system is controlled by `GLOBAL_CONFIG["Workers"]` and `GLOBAL_CONFIG["UseGPU"]` parameters.

## Current Configuration Analysis

### Global Configuration Parameters
- **Workers**: Currently set to `0` in `dea_config.json` (no parallelization)
- **UseGPU**: Currently set to `false` (GPU disabled)
- **Current CPU Assignment**: `os.cpu_count() - 1` (runtime override)

### Major Performance Bottlenecks Identified

## 1. Graph Matching Algorithm (GMA) - **CRITICAL BOTTLENECK**

**Location**: `graphMatching/gma.py` (`run_gma` function)
**Current State**: Sequential processing, no parallelization
**Impact**: This is likely the most CPU-intensive component

### Issues:
- Large similarity matrix computations (Alice vs Eve data)
- Sequential encoding operations  
- No parallel graph embedding generation
- Single-threaded matching algorithms

### Optimization Recommendations:
```python
# Recommended configuration for 19 CPUs
GLOBAL_CONFIG["Workers"] = 18  # Leave 1 CPU for system processes
```

**Specific Optimizations:**
1. **Parallel Similarity Matrix Computation**: 
   - Chunk similarity calculations across multiple processes
   - Use numpy's parallel BLAS operations
   
2. **Parallel Encoding**:
   - Split datasets across workers for Bloom Filter/TabMinHash encoding
   - Use Ray or multiprocessing for distributed encoding

3. **Parallel Graph Embedding**:
   - Node2Vec walks can be parallelized
   - NetMF matrix operations can leverage multiple cores

## 2. Neural Network Training - **GPU OPTIMIZATION OPPORTUNITY**

**Location**: `main.py` lines 200-400 (hyperparameter optimization and training)
**Current State**: Limited GPU utilization, suboptimal DataLoader configuration

### Issues:
- `num_workers=2` in DataLoaders (severely underutilized)
- Ray Tune using only 1 CPU per trial
- Sequential hyperparameter search
- Missing GPU memory optimizations

### Optimization Recommendations:

#### Enable GPU Processing:
```json
{
  "GLOBAL_CONFIG": {
    "UseGPU": true,
    "Workers": 18
  }
}
```

#### Optimize DataLoaders:
```python
# Current (suboptimal)
num_workers=2 if torch.cuda.is_available() else 0

# Optimized for 19 CPUs
num_workers=min(8, GLOBAL_CONFIG["Workers"]//2)  # 8 workers for data loading
pin_memory=True  # Enable for GPU
persistent_workers=True  # Keep workers alive between epochs
```

#### Parallel Hyperparameter Optimization:
```python
# Current (sequential)
max_concurrent_trials=GLOBAL_CONFIG["Workers"]  # Uses all CPUs for trials

# Optimized (balanced)
max_concurrent_trials=min(4, GLOBAL_CONFIG["Workers"]//4)  # 4 concurrent trials
# Each trial gets 4-5 CPUs for internal operations
```

## 3. Data Reconstruction - **ALREADY PARTIALLY OPTIMIZED**

**Location**: `utils.py` lines 630-680 (`fuzzy_reconstruction_approach`)
**Current State**: Well-implemented with joblib.Parallel
**Performance**: Good, but can be enhanced

### Current Implementation:
```python
reconstructed = Parallel(n_jobs=workers, batch_size=batch_size, prefer="processes")(
    delayed(reconstruct_fn)(entry) for entry in result
)
```

### Enhancement Recommendations:
1. **Dynamic Batch Sizing**: Already implemented, but could be tuned further
2. **Memory Management**: Consider using `backend='threading'` for I/O-bound operations
3. **Progress Monitoring**: Add progress bars for long-running reconstructions

## 4. Data Loading and Caching - **I/O OPTIMIZATION**

**Location**: `main.py` lines 85-140 (data loading functions)
**Current State**: Sequential file I/O, basic caching

### Issues:
- Sequential DataFrame loading
- No parallel data preprocessing
- Cache misses cause expensive recomputation

### Optimization Recommendations:

#### Parallel Data Loading:
```python
def load_data_parallel(data_directory, alice_enc_hash, identifier, load_test=False, workers=4):
    """Load multiple datasets in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            'reidentified': executor.submit(load_dataframe, 
                f"{data_directory}/available_to_eve/reidentified_individuals_{identifier}.h5"),
            'not_reidentified': executor.submit(load_dataframe, 
                f"{data_directory}/available_to_eve/not_reidentified_individuals_{identifier}.h5") if load_test else None,
            'all_data': executor.submit(load_dataframe, 
                f"{data_directory}/dev/alice_data_complete_with_encoding_{alice_enc_hash}.h5") if load_test else None
        }
        
        # Collect results
        results = {}
        for key, future in futures.items():
            if future is not None:
                results[key] = future.result()
        
        return results
```

## 5. Memory Optimization

### Current Issues:
- Large datasets loaded into memory simultaneously
- No memory-mapped file access
- Inefficient tensor operations

### Recommendations:

#### Memory-Mapped Dataset Loading:
```python
# For large datasets, use memory mapping
df = pd.read_hdf(filepath, mode='r')  # Read-only mode
# Consider using Dask for out-of-core processing
```

#### GPU Memory Optimization:
```python
# Add to model training
torch.cuda.empty_cache()  # Clear GPU cache between trials
model.half()  # Use FP16 for inference if possible
```

## Specific Configuration Recommendations

### Recommended `dea_config.json` Changes:

```json
{
    "GLOBAL_CONFIG": {
        "Workers": 18,
        "UseGPU": true,
        "Verbose": true,
        "BenchMode": true
    }
}
```

### Ray Configuration Optimization:

```python
# Current
ray.init(
    num_cpus=GLOBAL_CONFIG["Workers"],
    num_gpus=1 if GLOBAL_CONFIG["UseGPU"] else 0,
    ignore_reinit_error=True,
    logging_level="ERROR"
)

# Optimized
ray.init(
    num_cpus=GLOBAL_CONFIG["Workers"],
    num_gpus=1 if GLOBAL_CONFIG["UseGPU"] else 0,
    object_store_memory=2000000000,  # 2GB object store
    ignore_reinit_error=True,
    logging_level="ERROR"
)
```

## Implementation Priority

### High Priority (Immediate Impact):
1. ✅ **Update GLOBAL_CONFIG**: Set `Workers: 18` and `UseGPU: true`
2. ✅ **Optimize DataLoader num_workers**: Increase from 2 to 8
3. ✅ **Parallel GMA encoding**: Implement chunked processing in graph matching

### Medium Priority (Significant Impact):
4. ✅ **Ray Tune optimization**: Reduce concurrent trials, increase per-trial resources
5. ✅ **Memory management**: Add GPU cache clearing and memory mapping
6. ✅ **Parallel data loading**: Implement concurrent file I/O

### Low Priority (Optimization):
7. ✅ **Advanced GPU features**: Mixed precision training, gradient accumulation
8. ✅ **Monitoring**: Add detailed performance profiling
9. ✅ **Caching improvements**: Implement more aggressive caching strategies

## Expected Performance Improvements

### With Optimizations:
- **Graph Matching**: 10-15x speedup (from sequential to 18-core parallel)
- **Neural Network Training**: 2-3x speedup (GPU utilization + optimized data loading)
- **Data Reconstruction**: 1.5-2x speedup (optimized batch sizes and backend selection)
- **Overall Pipeline**: 5-8x speedup expected

### Resource Utilization:
- **CPU**: 85-90% utilization across 18 cores
- **GPU**: 70-80% utilization during training phases
- **Memory**: More efficient usage with memory mapping and caching
- **I/O**: Reduced bottlenecks through parallel loading

## Monitoring and Profiling

### Add Performance Monitoring:
```python
import psutil
import GPUtil

def log_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    memory = psutil.virtual_memory()
    if GLOBAL_CONFIG["UseGPU"]:
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load if gpus else 0
    
    print(f"CPU Usage: {np.mean(cpu_percent):.1f}% (per core: {cpu_percent})")
    print(f"Memory: {memory.percent}% ({memory.used//1024//1024}MB used)")
    if GLOBAL_CONFIG["UseGPU"]:
        print(f"GPU Usage: {gpu_usage*100:.1f}%")
```

This optimization plan should significantly improve performance while maintaining code stability and correctness. The key is to leverage the 19 CPUs and 1 GPU effectively through proper parallel processing configuration.