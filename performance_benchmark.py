#!/usr/bin/env python3
"""
Performance Benchmarking Script for ML Pipeline Optimizations

This script benchmarks the performance improvements from various optimizations:
- Data loading performance
- Training speed with/without optimizations
- Memory usage
- Model compilation impact
"""

import time
import torch
import psutil
import gc
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

# Import original and optimized components
from datasets.bloom_filter_dataset import BloomFilterDataset
from datasets.optimized_bloom_filter_dataset import OptimizedBloomFilterDataset
from pytorch_models.base_model import BaseModel
from pytorch_models.optimized_base_model import OptimizedBaseModel
from utils import run_epoch, create_optimized_dataloader, load_dataframe

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for ML pipeline optimizations.
    """
    
    def __init__(self, data_path=None, output_dir="benchmark_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔬 Performance Benchmark initialized")
        print(f"Device: {self.device}")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    @contextmanager
    def timer(self, operation_name):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        yield
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        result = {
            'operation': operation_name,
            'duration_seconds': duration,
            'memory_delta_mb': memory_delta,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        print(f"✅ {operation_name}: {duration:.2f}s, Memory Δ: {memory_delta:+.1f}MB")

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process().memory_info().rss / (1024 * 1024)

    def benchmark_dataloader_performance(self, dataset, batch_sizes=[16, 32, 64, 128]):
        """
        Benchmark DataLoader performance with different configurations.
        """
        print("\n📊 Benchmarking DataLoader Performance...")
        
        dataloader_results = []
        
        for batch_size in batch_sizes:
            # Original DataLoader
            with self.timer(f"Original DataLoader (batch_size={batch_size})"):
                original_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=2
                )
                self._iterate_dataloader(original_loader, max_batches=10)
            
            # Optimized DataLoader
            with self.timer(f"Optimized DataLoader (batch_size={batch_size})"):
                optimized_loader = create_optimized_dataloader(
                    dataset,
                    batch_size=batch_size,
                    is_training=True
                )
                self._iterate_dataloader(optimized_loader, max_batches=10)

    def _iterate_dataloader(self, dataloader, max_batches=10):
        """Helper to iterate through dataloader for benchmarking."""
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            # Move data to device to simulate real training
            if len(batch) == 3:
                data, labels, _ = batch
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                data, _ = batch
                data = data.to(self.device, non_blocking=True)

    def benchmark_model_performance(self, input_dim=1000, output_dim=676, hidden_layer=128):
        """
        Benchmark model training performance.
        """
        print("\n🏋️ Benchmarking Model Performance...")
        
        # Create synthetic data for consistent benchmarking
        num_samples = 1000
        synthetic_data = torch.randn(num_samples, input_dim)
        synthetic_labels = torch.randint(0, 2, (num_samples, output_dim)).float()
        
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_labels)
        
        # Benchmark original model
        with self.timer("Original Model Training (5 epochs)"):
            original_model = BaseModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layer=hidden_layer,
                num_layers=2,
                dropout_rate=0.2,
                activation_fn="relu"
            ).to(self.device)
            
            self._train_model_benchmark(original_model, synthetic_dataset, epochs=5)
        
        # Benchmark optimized model
        with self.timer("Optimized Model Training (5 epochs)"):
            optimized_model = OptimizedBaseModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layer=hidden_layer,
                num_layers=2,
                dropout_rate=0.2,
                activation_fn="relu",
                compile_model=True
            ).to(self.device)
            
            self._train_model_benchmark(optimized_model, synthetic_dataset, epochs=5)

    def _train_model_benchmark(self, model, dataset, epochs=5, batch_size=32):
        """Helper to train model for benchmarking."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            loss = run_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                is_training=True,
                verbose=False,
                accumulation_steps=1
            )

    def benchmark_memory_optimization(self):
        """
        Benchmark memory optimization techniques.
        """
        print("\n🧠 Benchmarking Memory Optimizations...")
        
        # Test different batch sizes and accumulation steps
        input_dim, output_dim = 1000, 676
        num_samples = 2000
        
        synthetic_data = torch.randn(num_samples, input_dim)
        synthetic_labels = torch.randint(0, 2, (num_samples, output_dim)).float()
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_data, synthetic_labels)
        
        model = OptimizedBaseModel(input_dim, output_dim).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test different configurations
        configs = [
            {"batch_size": 64, "accumulation_steps": 1, "name": "Standard Training"},
            {"batch_size": 32, "accumulation_steps": 2, "name": "Gradient Accumulation (2x)"},
            {"batch_size": 16, "accumulation_steps": 4, "name": "Gradient Accumulation (4x)"},
        ]
        
        for config in configs:
            with self.timer(config["name"]):
                dataloader = DataLoader(
                    synthetic_dataset, 
                    batch_size=config["batch_size"], 
                    shuffle=True
                )
                
                loss = run_epoch(
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    is_training=True,
                    verbose=False,
                    accumulation_steps=config["accumulation_steps"]
                )

    def generate_report(self):
        """
        Generate a comprehensive performance report.
        """
        if not self.results:
            print("⚠️  No benchmark results to report")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        df.to_csv(f"{self.output_dir}/benchmark_results.csv", index=False)
        
        # Generate summary statistics
        summary = df.groupby('operation').agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'memory_delta_mb': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print("\n📈 Performance Benchmark Summary")
        print("=" * 80)
        print(summary)
        
        # Create visualizations
        self._create_visualizations(df)
        
        # Calculate optimization impact
        self._calculate_optimization_impact(df)

    def _create_visualizations(self, df):
        """Create performance visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Duration comparison
        duration_data = df.pivot_table(
            index='operation', 
            values='duration_seconds', 
            aggfunc='mean'
        ).sort_values('duration_seconds')
        
        duration_data.plot(kind='barh', ax=axes[0, 0])
        axes[0, 0].set_title('Average Operation Duration')
        axes[0, 0].set_xlabel('Duration (seconds)')
        
        # Memory usage comparison
        memory_data = df.pivot_table(
            index='operation', 
            values='memory_delta_mb', 
            aggfunc='mean'
        ).sort_values('memory_delta_mb')
        
        memory_data.plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Average Memory Delta')
        axes[0, 1].set_xlabel('Memory Delta (MB)')
        
        # Timeline plot
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        for i, operation in enumerate(df['operation'].unique()):
            op_data = df[df['operation'] == operation]
            axes[1, 0].plot(range(len(op_data)), op_data['duration_seconds'], 
                           marker='o', label=operation[:20])
        
        axes[1, 0].set_title('Performance Over Time')
        axes[1, 0].set_xlabel('Run Number')
        axes[1, 0].set_ylabel('Duration (seconds)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Memory efficiency scatter
        axes[1, 1].scatter(df['duration_seconds'], df['memory_delta_mb'], 
                          alpha=0.6, c=range(len(df)), cmap='viridis')
        axes[1, 1].set_xlabel('Duration (seconds)')
        axes[1, 1].set_ylabel('Memory Delta (MB)')
        axes[1, 1].set_title('Duration vs Memory Usage')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_optimization_impact(self, df):
        """Calculate the impact of optimizations."""
        print("\n🚀 Optimization Impact Analysis")
        print("=" * 50)
        
        # Compare original vs optimized operations
        comparisons = [
            ("Original DataLoader", "Optimized DataLoader"),
            ("Original Model Training", "Optimized Model Training"),
        ]
        
        for original, optimized in comparisons:
            orig_ops = df[df['operation'].str.contains(original, na=False)]
            opt_ops = df[df['operation'].str.contains(optimized, na=False)]
            
            if not orig_ops.empty and not opt_ops.empty:
                orig_time = orig_ops['duration_seconds'].mean()
                opt_time = opt_ops['duration_seconds'].mean()
                
                improvement = ((orig_time - opt_time) / orig_time) * 100
                
                print(f"{original} vs {optimized}:")
                print(f"  Original: {orig_time:.2f}s")
                print(f"  Optimized: {opt_time:.2f}s")
                print(f"  Improvement: {improvement:+.1f}%")
                print()

def main():
    """
    Main benchmarking function.
    """
    benchmark = PerformanceBenchmark()
    
    print("🚀 Starting Performance Benchmark Suite...")
    print("This will test various optimizations and measure their impact.")
    
    try:
        # Create a small synthetic dataset for testing
        print("\n📦 Creating synthetic dataset for benchmarking...")
        
        # Create minimal synthetic data
        num_samples = 1000
        bloom_filter_data = [''.join(np.random.choice(['0', '1'], 100)) for _ in range(num_samples)]
        uids = [f"uid_{i}" for i in range(num_samples)]
        
        synthetic_df = pd.DataFrame({
            'bloomfilter': bloom_filter_data,
            'uid': uids
        })
        
        # Create alphabet for two-grams
        import string
        alphabet = string.ascii_lowercase
        digits = string.digits
        all_two_grams = [a + b for a in alphabet for b in alphabet]
        
        # Create dataset
        dataset = BloomFilterDataset(
            synthetic_df, 
            is_labeled=False, 
            all_two_grams=all_two_grams
        )
        
        # Run benchmarks
        benchmark.benchmark_dataloader_performance(dataset)
        benchmark.benchmark_model_performance()
        benchmark.benchmark_memory_optimization()
        
        # Generate comprehensive report
        benchmark.generate_report()
        
        print(f"\n✅ Benchmarking complete! Results saved to {benchmark.output_dir}/")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()