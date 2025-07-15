#!/usr/bin/env python3
"""
Optimized Data Caching System

This module provides intelligent caching mechanisms to eliminate redundant data loading
and dramatically improve startup times for the ML pipeline.
"""

import os
import pickle
import hashlib
import time
from typing import Optional, Tuple, Any
from functools import wraps
import pandas as pd
import hickle as hkl
import torch
from torch.utils.data import Dataset

class SmartDataCache:
    """
    Intelligent data caching system with automatic invalidation and optimization.
    """
    
    def __init__(self, cache_dir="data/cache", max_cache_size_gb=5.0, compression=True):
        self.cache_dir = cache_dir
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = os.path.join(cache_dir, "cache_metadata.pkl")
        self.metadata = self._load_metadata()
        
        print(f"🗄️  Smart Data Cache initialized at {cache_dir}")
        self._cleanup_old_cache()

    def _load_metadata(self):
        """Load cache metadata or create new if doesn't exist."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load cache metadata: {e}")
        
        return {
            'entries': {},
            'access_times': {},
            'file_sizes': {},
            'created': time.time()
        }

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"⚠️  Failed to save cache metadata: {e}")

    def _generate_cache_key(self, *args, **kwargs):
        """Generate a unique cache key from arguments."""
        # Convert arguments to a hashable string
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key, suffix=".pkl"):
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"cache_{cache_key}{suffix}")

    def _cleanup_old_cache(self):
        """Remove old cache entries to maintain size limits."""
        total_size = sum(self.metadata['file_sizes'].values())
        max_size_bytes = self.max_cache_size_gb * 1024 * 1024 * 1024
        
        if total_size > max_size_bytes:
            print(f"🧹 Cache size ({total_size / (1024**3):.1f}GB) exceeds limit, cleaning up...")
            
            # Sort by access time (least recently used first)
            entries_by_access = sorted(
                self.metadata['access_times'].items(),
                key=lambda x: x[1]
            )
            
            for cache_key, _ in entries_by_access:
                if total_size <= max_size_bytes:
                    break
                
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    file_size = self.metadata['file_sizes'].get(cache_key, 0)
                    os.remove(cache_path)
                    total_size -= file_size
                    
                    # Remove from metadata
                    for metadata_dict in self.metadata.values():
                        if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                            del metadata_dict[cache_key]
            
            self._save_metadata()
            print(f"✅ Cache cleaned up, new size: {total_size / (1024**3):.1f}GB")

    def cache_dataset(self, cache_key: str, dataset: Dataset, force_refresh=False):
        """
        Cache a dataset with intelligent compression and metadata tracking.
        """
        cache_path = self._get_cache_path(cache_key, ".pkl")
        
        if not force_refresh and os.path.exists(cache_path):
            # Update access time
            self.metadata['access_times'][cache_key] = time.time()
            self._save_metadata()
            print(f"📖 Loading cached dataset: {cache_key}")
            return self.load_cached_dataset(cache_key)
        
        print(f"💾 Caching dataset: {cache_key}")
        
        # Cache the dataset
        try:
            # Extract dataset components for efficient storage
            cache_data = {
                'class_name': dataset.__class__.__name__,
                'attributes': {
                    attr: getattr(dataset, attr)
                    for attr in dir(dataset)
                    if not attr.startswith('_') and not callable(getattr(dataset, attr))
                }
            }
            
            # Use compression if enabled
            if self.compression:
                import lz4.frame
                serialized_data = pickle.dumps(cache_data)
                compressed_data = lz4.frame.compress(serialized_data)
                
                with open(cache_path, 'wb') as f:
                    f.write(compressed_data)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            # Update metadata
            file_size = os.path.getsize(cache_path)
            self.metadata['entries'][cache_key] = {
                'file_path': cache_path,
                'created': time.time(),
                'compressed': self.compression
            }
            self.metadata['access_times'][cache_key] = time.time()
            self.metadata['file_sizes'][cache_key] = file_size
            
            self._save_metadata()
            print(f"✅ Dataset cached successfully ({file_size / (1024*1024):.1f}MB)")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to cache dataset: {e}")
            return dataset

    def load_cached_dataset(self, cache_key: str) -> Optional[Dataset]:
        """
        Load a cached dataset if it exists.
        """
        if cache_key not in self.metadata['entries']:
            return None
        
        cache_path = self._get_cache_path(cache_key, ".pkl")
        
        if not os.path.exists(cache_path):
            # Remove from metadata if file doesn't exist
            for metadata_dict in self.metadata.values():
                if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                    del metadata_dict[cache_key]
            self._save_metadata()
            return None
        
        try:
            # Load cached data
            with open(cache_path, 'rb') as f:
                if self.metadata['entries'][cache_key].get('compressed', False):
                    import lz4.frame
                    compressed_data = f.read()
                    serialized_data = lz4.frame.decompress(compressed_data)
                    cache_data = pickle.loads(serialized_data)
                else:
                    cache_data = pickle.load(f)
            
            # Update access time
            self.metadata['access_times'][cache_key] = time.time()
            self._save_metadata()
            
            print(f"📂 Loaded cached dataset: {cache_key}")
            return cache_data
            
        except Exception as e:
            print(f"❌ Failed to load cached dataset: {e}")
            return None

    def cache_dataframe(self, cache_key: str, df: pd.DataFrame, force_refresh=False):
        """
        Cache a pandas DataFrame with optimized storage.
        """
        cache_path = self._get_cache_path(cache_key, ".h5")
        
        if not force_refresh and os.path.exists(cache_path):
            self.metadata['access_times'][cache_key] = time.time()
            self._save_metadata()
            return self.load_cached_dataframe(cache_key)
        
        print(f"💾 Caching DataFrame: {cache_key}")
        
        try:
            # Use HDF5 for efficient DataFrame storage
            df.to_hdf(cache_path, key='data', mode='w', complevel=9, complib='zlib')
            
            # Update metadata
            file_size = os.path.getsize(cache_path)
            self.metadata['entries'][cache_key] = {
                'file_path': cache_path,
                'created': time.time(),
                'type': 'dataframe'
            }
            self.metadata['access_times'][cache_key] = time.time()
            self.metadata['file_sizes'][cache_key] = file_size
            
            self._save_metadata()
            print(f"✅ DataFrame cached successfully ({file_size / (1024*1024):.1f}MB)")
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to cache DataFrame: {e}")
            return df

    def load_cached_dataframe(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load a cached DataFrame.
        """
        if cache_key not in self.metadata['entries']:
            return None
        
        cache_path = self._get_cache_path(cache_key, ".h5")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            df = pd.read_hdf(cache_path, key='data')
            
            # Update access time
            self.metadata['access_times'][cache_key] = time.time()
            self._save_metadata()
            
            print(f"📂 Loaded cached DataFrame: {cache_key}")
            return df
            
        except Exception as e:
            print(f"❌ Failed to load cached DataFrame: {e}")
            return None

    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear cache entries, optionally matching a pattern.
        """
        if pattern is None:
            # Clear all cache
            for cache_key in list(self.metadata['entries'].keys()):
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            
            self.metadata = {
                'entries': {},
                'access_times': {},
                'file_sizes': {},
                'created': time.time()
            }
            self._save_metadata()
            print("🗑️  All cache cleared")
        else:
            # Clear cache entries matching pattern
            to_remove = [key for key in self.metadata['entries'].keys() if pattern in key]
            
            for cache_key in to_remove:
                cache_path = self._get_cache_path(cache_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                # Remove from metadata
                for metadata_dict in self.metadata.values():
                    if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                        del metadata_dict[cache_key]
            
            self._save_metadata()
            print(f"🗑️  Cleared {len(to_remove)} cache entries matching '{pattern}'")

    def get_cache_stats(self):
        """
        Get statistics about the cache.
        """
        total_size = sum(self.metadata['file_sizes'].values())
        num_entries = len(self.metadata['entries'])
        
        stats = {
            'num_entries': num_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'average_size_mb': (total_size / num_entries) / (1024 * 1024) if num_entries > 0 else 0,
            'cache_dir': self.cache_dir,
            'entries': list(self.metadata['entries'].keys())
        }
        
        return stats

def cached_dataloader(cache_instance: SmartDataCache):
    """
    Decorator for caching data loading operations.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = cache_instance._generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to load from cache
            cached_result = cache_instance.load_cached_dataset(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if isinstance(result, (Dataset, pd.DataFrame)):
                cache_instance.cache_dataset(cache_key, result)
            
            return result
        return wrapper
    return decorator

# Global cache instance
_global_cache = None

def get_global_cache():
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartDataCache()
    return _global_cache

def optimized_load_dataframe(path: str, use_cache=True):
    """
    Optimized dataframe loading with automatic caching.
    """
    if not use_cache:
        # Load directly without caching
        data = hkl.load(path)
        return pd.DataFrame(data[1:], columns=data[0])
    
    cache = get_global_cache()
    cache_key = f"df_{hashlib.md5(path.encode()).hexdigest()}"
    
    # Try to load from cache
    cached_df = cache.load_cached_dataframe(cache_key)
    if cached_df is not None:
        return cached_df
    
    # Load and cache
    print(f"📥 Loading DataFrame from {path}")
    data = hkl.load(path)
    df = pd.DataFrame(data[1:], columns=data[0])
    
    return cache.cache_dataframe(cache_key, df)

if __name__ == "__main__":
    # Test the caching system
    cache = SmartDataCache()
    
    # Display cache statistics
    stats = cache.get_cache_stats()
    print("\n📊 Cache Statistics:")
    for key, value in stats.items():
        if key != 'entries':
            print(f"  {key}: {value}")
    
    if stats['entries']:
        print("  Cached entries:")
        for entry in stats['entries']:
            print(f"    - {entry}")