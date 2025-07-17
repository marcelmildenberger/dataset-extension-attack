#!/usr/bin/env python3
"""
Improved Dataset Extension Attack (DEA) Analysis Script

This script provides comprehensive analysis and visualization of DEA experimental results,
examining privacy attack effectiveness across different encoding schemes, datasets,
and configurations.

Author: Assistant (Refactored and Enhanced)
Date: January 2025
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import re
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration constants
RESULTS_FILE = "formatted_results.csv"
OUTPUT_DIR = Path("analysis_improved")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
SUMMARY_DIR = OUTPUT_DIR / "summaries"

# Color schemes for consistent visualization
ENCODING_COLORS = {
    'BloomFilter': '#2E86AB',    # Blue
    'TabMinHash': '#A23B72',     # Pink/Purple  
    'TwoStepHash': '#F18F01'     # Orange
}

DROP_COLORS = {
    'Eve': '#E63946',    # Red
    'Both': '#457B9D'    # Blue
}

DATASET_COLORS = {
    'fakename_1k': '#264653',
    'fakename_2k': '#2A9D8F', 
    'fakename_5k': '#E9C46A',
    'fakename_10k': '#F4A261',
    'fakename_20k': '#E76F51',
    'fakename_50k': '#A8DADC',
    'euro_person': '#1D3557'
}

# Baseline performance metrics for comparison
BASELINE_METRICS = {
    "fakename_1k":   {"Precision": 0.2162, "Recall": 0.2476, "F1": 0.2300},
    "fakename_2k":   {"Precision": 0.2131, "Recall": 0.2452, "F1": 0.2271},
    "fakename_5k":   {"Precision": 0.2144, "Recall": 0.2470, "F1": 0.2287},
    "fakename_10k":  {"Precision": 0.2151, "Recall": 0.2467, "F1": 0.2289},
    "fakename_20k":  {"Precision": 0.2153, "Recall": 0.2473, "F1": 0.2293},
    "fakename_50k":  {"Precision": 0.2151, "Recall": 0.2463, "F1": 0.2288},
    "titanic_full":  {"Precision": 0.2468, "Recall": 0.3770, "F1": 0.2896},
    "euro_person":   {"Precision": 0.2197, "Recall": 0.2446, "F1": 0.2306}
}


class DEAAnalyzer:
    """
    Dataset Extension Attack Analysis Class
    
    Provides comprehensive analysis capabilities for DEA experimental results,
    including data preprocessing, statistical analysis, and visualization generation.
    """
    
    def __init__(self, results_file: str = RESULTS_FILE):
        """Initialize the analyzer with experimental results."""
        self.df = self._load_and_preprocess_data(results_file)
        self._setup_output_directories()
        self._setup_plot_style()
        
    def _load_and_preprocess_data(self, results_file: str) -> pd.DataFrame:
        """Load and preprocess the experimental results data."""
        print("📊 Loading and preprocessing experimental data...")
        
        df = pd.read_csv(results_file)
        
        # Extract dataset size information
        def extract_dataset_size(name: str) -> Optional[int]:
            """Extract numeric dataset size from dataset name."""
            if "euro" in name.lower():
                return 25000  # Estimated based on context
            match = re.search(r"(\d+)k", name)
            return int(match.group(1)) * 1000 if match else None
        
        # Data preprocessing
        df["DatasetSize"] = df["Dataset"].apply(extract_dataset_size)
        df["DatasetClean"] = df["Dataset"].str.replace(".tsv", "").str.replace("_", " ")
        df["Overlap"] = df["Overlap"].astype(float)
        
        # Handle missing values in runtime columns
        time_cols = [
            "GraphMatchingAttackTime", "HyperparameterOptimizationTime",
            "ModelTrainingTime", "ApplicationtoEncodedDataTime", 
            "RefinementandReconstructionTime", "TotalRuntime"
        ]
        df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce")
        
        # Calculate derived metrics
        df["ReidentificationRatePercent"] = df["ReidentificationRate"] * 100
        df["ReidentificationRateFuzzyPercent"] = df["ReidentificationRateFuzzy"] * 100
        df["ReidentificationRateGreedyPercent"] = df["ReidentificationRateGreedy"] * 100
        
        # Performance improvement over baseline
        df["F1_Improvement"] = df.apply(self._calculate_f1_improvement, axis=1)
        
        print(f"✅ Loaded {len(df)} experimental results across {df['Encoding'].nunique()} encoding schemes")
        return df
    
    def _calculate_f1_improvement(self, row) -> Optional[float]:
        """Calculate F1 improvement over baseline for each row."""
        dataset_key = row["Dataset"].replace(".tsv", "")
        if dataset_key in BASELINE_METRICS:
            baseline_f1 = BASELINE_METRICS[dataset_key]["F1"]
            return row["TrainedF1"] - baseline_f1
        return None
    
    def _setup_output_directories(self):
        """Create output directories for analysis results."""
        for directory in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, SUMMARY_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directories created at {OUTPUT_DIR}")
    
    def _setup_plot_style(self):
        """Configure matplotlib and seaborn styling for consistent plots."""
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        # Custom color palette
        custom_palette = list(ENCODING_COLORS.values())
        sns.set_palette(custom_palette)
        
        # Global plot parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def generate_executive_summary(self) -> Dict:
        """Generate high-level summary statistics for the analysis."""
        print("📈 Generating executive summary...")
        
        summary = {
            'total_experiments': len(self.df),
            'encoding_schemes': self.df['Encoding'].nunique(),
            'datasets_tested': self.df['Dataset'].nunique(),
            'avg_reidentification_rate': self.df['ReidentificationRatePercent'].mean(),
            'max_reidentification_rate': self.df['ReidentificationRatePercent'].max(),
            'avg_f1_score': self.df['TrainedF1'].mean(),
            'max_f1_score': self.df['TrainedF1'].max(),
            'avg_total_runtime': self.df['TotalRuntime'].mean(),
            'encoding_performance': {},
            'dataset_vulnerability': {}
        }
        
        # Performance by encoding scheme
        for encoding in self.df['Encoding'].unique():
            encoding_data = self.df[self.df['Encoding'] == encoding]
            summary['encoding_performance'][encoding] = {
                'avg_reidentification_rate': encoding_data['ReidentificationRatePercent'].mean(),
                'avg_f1_score': encoding_data['TrainedF1'].mean(),
                'experiment_count': len(encoding_data)
            }
        
        # Vulnerability by dataset size
        for dataset in self.df['Dataset'].unique():
            dataset_data = self.df[self.df['Dataset'] == dataset]
            summary['dataset_vulnerability'][dataset] = {
                'avg_reidentification_rate': dataset_data['ReidentificationRatePercent'].mean(),
                'max_reidentification_rate': dataset_data['ReidentificationRatePercent'].max(),
                'size': dataset_data['DatasetSize'].iloc[0] if not dataset_data['DatasetSize'].isna().all() else None
            }
        
        # Save summary to file
        summary_file = SUMMARY_DIR / "executive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("DATASET EXTENSION ATTACK (DEA) - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Experiments: {summary['total_experiments']}\n")
            f.write(f"Encoding Schemes Tested: {summary['encoding_schemes']}\n")
            f.write(f"Datasets Analyzed: {summary['datasets_tested']}\n")
            f.write(f"Average Re-identification Rate: {summary['avg_reidentification_rate']:.2f}%\n")
            f.write(f"Maximum Re-identification Rate: {summary['max_reidentification_rate']:.2f}%\n")
            f.write(f"Average F1 Score: {summary['avg_f1_score']:.3f}\n")
            f.write(f"Maximum F1 Score: {summary['max_f1_score']:.3f}\n")
            f.write(f"Average Total Runtime: {summary['avg_total_runtime']:.1f} minutes\n\n")
            
            f.write("ENCODING SCHEME PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for encoding, metrics in summary['encoding_performance'].items():
                f.write(f"{encoding}:\n")
                f.write(f"  - Avg Re-ID Rate: {metrics['avg_reidentification_rate']:.2f}%\n")
                f.write(f"  - Avg F1 Score: {metrics['avg_f1_score']:.3f}\n")
                f.write(f"  - Experiments: {metrics['experiment_count']}\n\n")
        
        return summary
    
    def plot_encoding_comparison_comprehensive(self):
        """Create comprehensive comparison of encoding schemes across all metrics."""
        print("🎨 Creating comprehensive encoding comparison plots...")
        
        # Set up the subplot grid
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define metrics to plot
        metrics = [
            ('TrainedF1', 'F1 Score', 'Performance'),
            ('ReidentificationRatePercent', 'Re-identification Rate (%)', 'Privacy Risk'),
            ('TotalRuntime', 'Total Runtime (min)', 'Computational Cost'),
            ('TrainedPrecision', 'Precision', 'Classification Quality'),
            ('TrainedRecall', 'Recall', 'Classification Quality'),
            ('HyperparameterOptimizationTime', 'Hyperparameter Optimization (min)', 'Setup Time')
        ]
        
        # Create individual metric comparisons
        for i, (metric, title, category) in enumerate(metrics):
            if i >= 6:  # Limit to 6 plots
                break
                
            row, col = divmod(i, 3)
            ax = fig.add_subplot(gs[row, col])
            
            # Aggregate data by encoding
            grouped_data = self.df.groupby('Encoding')[metric].agg(['mean', 'std', 'count']).reset_index()
            
            # Create bar plot with error bars
            bars = ax.bar(grouped_data['Encoding'], grouped_data['mean'], 
                         yerr=grouped_data['std'], capsize=5,
                         color=[ENCODING_COLORS[enc] for enc in grouped_data['Encoding']],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_title(f'{title} by Encoding Scheme', fontweight='bold', pad=15)
            ax.set_ylabel(title)
            ax.set_xlabel('Encoding Scheme')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean_val, count in zip(bars, grouped_data['mean'], grouped_data['count']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'n={count}', ha='center', va='center', fontsize=7, 
                       color='white', fontweight='bold')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
        
        # Add overall title
        fig.suptitle('Dataset Extension Attack: Comprehensive Encoding Scheme Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plt.savefig(PLOTS_DIR / "encoding_scheme_comprehensive_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_overlap_impact_analysis(self):
        """Analyze the impact of overlap percentage on attack success."""
        print("🎨 Creating overlap impact analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define metrics for overlap analysis
        overlap_metrics = [
            ('ReidentificationRatePercent', 'Re-identification Rate (%)', 'Privacy Risk'),
            ('TrainedF1', 'F1 Score', 'Model Performance'),
            ('TotalRuntime', 'Total Runtime (min)', 'Computational Cost'),
            ('HyperparameterOptimizationTime', 'Hyperparameter Optimization Time (min)', 'Setup Cost')
        ]
        
        for i, (metric, title, category) in enumerate(overlap_metrics):
            ax = axes[i]
            
            # Plot for each encoding scheme
            for encoding in self.df['Encoding'].unique():
                encoding_data = self.df[self.df['Encoding'] == encoding]
                
                # Group by overlap and calculate mean
                overlap_grouped = encoding_data.groupby('Overlap')[metric].mean().reset_index()
                
                # Plot line with markers
                ax.plot(overlap_grouped['Overlap'], overlap_grouped[metric], 
                       marker='o', linewidth=2, markersize=6, 
                       label=encoding, color=ENCODING_COLORS[encoding])
            
            ax.set_title(f'{title} vs. Overlap Percentage', fontweight='bold')
            ax.set_xlabel('Overlap Percentage')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0.2, 0.4, 0.6, 0.8])
            ax.set_xticklabels(['20%', '40%', '60%', '80%'])
        
        plt.suptitle('Impact of Data Overlap on DEA Performance', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "overlap_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_scalability_analysis(self):
        """Analyze how attack performance scales with dataset size."""
        print("🎨 Creating dataset scalability analysis...")
        
        # Filter out rows where DatasetSize is None
        scalability_data = self.df.dropna(subset=['DatasetSize'])
        
        if len(scalability_data) == 0:
            print("⚠️  No dataset size information available for scalability analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = [
            ('ReidentificationRatePercent', 'Re-identification Rate (%)'),
            ('TrainedF1', 'F1 Score'),
            ('TotalRuntime', 'Total Runtime (min)'),
            ('HyperparameterOptimizationTime', 'Hyperparameter Optimization Time (min)')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i]
            
            # Create scatter plot with trend lines for each encoding
            for encoding in scalability_data['Encoding'].unique():
                encoding_data = scalability_data[scalability_data['Encoding'] == encoding]
                
                if len(encoding_data) > 1:  # Need at least 2 points for trend line
                    # Scatter plot
                    ax.scatter(encoding_data['DatasetSize'], encoding_data[metric], 
                             alpha=0.6, s=50, label=encoding, 
                             color=ENCODING_COLORS[encoding])
                    
                    # Trend line
                    z = np.polyfit(encoding_data['DatasetSize'], encoding_data[metric], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(encoding_data['DatasetSize'].min(), 
                                        encoding_data['DatasetSize'].max(), 100)
                    ax.plot(x_trend, p(x_trend), "--", alpha=0.8, 
                           color=ENCODING_COLORS[encoding], linewidth=2)
            
            ax.set_title(f'{title} vs. Dataset Size', fontweight='bold')
            ax.set_xlabel('Dataset Size (records)')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')  # Log scale for dataset size
        
        plt.suptitle('Dataset Extension Attack: Scalability Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "dataset_scalability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_privacy_risk_heatmap(self):
        """Create a heatmap showing privacy risk across different configurations."""
        print("🎨 Creating privacy risk heatmap...")
        
        # Create pivot table for heatmap
        heatmap_data = self.df.pivot_table(
            values='ReidentificationRatePercent',
            index='Encoding',
            columns='Overlap',
            aggfunc='mean'
        )
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap with custom colormap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Reds', 
                   cbar_kws={'label': 'Re-identification Rate (%)'}, 
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Privacy Risk Heatmap: Re-identification Rates by Configuration', 
                    fontweight='bold', pad=20)
        ax.set_xlabel('Overlap Percentage')
        ax.set_ylabel('Encoding Scheme')
        
        # Format x-axis labels as percentages
        x_labels = [f'{int(float(label.get_text())*100)}%' for label in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "privacy_risk_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_vs_privacy_tradeoff(self):
        """Create scatter plot showing performance vs privacy tradeoff."""
        print("🎨 Creating performance vs privacy tradeoff analysis...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot for each encoding scheme
        for encoding in self.df['Encoding'].unique():
            encoding_data = self.df[self.df['Encoding'] == encoding]
            
            scatter = ax.scatter(encoding_data['TrainedF1'], 
                               encoding_data['ReidentificationRatePercent'],
                               alpha=0.7, s=60, label=encoding,
                               color=ENCODING_COLORS[encoding])
        
        # Add diagonal line to show ideal tradeoff
        ax.plot([0, 1], [0, 100], 'k--', alpha=0.3, linewidth=1, 
               label='Linear Tradeoff Reference')
        
        # Customize plot
        ax.set_xlabel('Model Performance (F1 Score)', fontweight='bold')
        ax.set_ylabel('Privacy Risk (Re-identification Rate %)', fontweight='bold')
        ax.set_title('Performance vs Privacy Tradeoff in Dataset Extension Attacks', 
                    fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax.text(0.05, 95, 'Low Performance\nHigh Privacy Risk', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
               fontsize=9, ha='left', va='top')
        
        ax.text(0.95, 5, 'High Performance\nLow Privacy Risk', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
               fontsize=9, ha='right', va='bottom')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "performance_privacy_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistical_summary_tables(self):
        """Generate comprehensive statistical summary tables."""
        print("📋 Generating statistical summary tables...")
        
        # Overall summary statistics
        summary_stats = self.df.groupby('Encoding').agg({
            'ReidentificationRatePercent': ['mean', 'std', 'min', 'max'],
            'TrainedF1': ['mean', 'std', 'min', 'max'],
            'TrainedPrecision': ['mean', 'std', 'min', 'max'],
            'TrainedRecall': ['mean', 'std', 'min', 'max'],
            'TotalRuntime': ['mean', 'std', 'min', 'max'],
            'HyperparameterOptimizationTime': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats.to_csv(TABLES_DIR / "encoding_summary_statistics.csv")
        
        # Overlap impact summary
        overlap_summary = self.df.groupby(['Encoding', 'Overlap']).agg({
            'ReidentificationRatePercent': 'mean',
            'TrainedF1': 'mean',
            'TotalRuntime': 'mean'
        }).round(4)
        overlap_summary.to_csv(TABLES_DIR / "overlap_impact_summary.csv")
        
        # Drop strategy comparison
        drop_strategy_summary = self.df.groupby(['Encoding', 'DropFrom']).agg({
            'ReidentificationRatePercent': ['mean', 'std', 'count'],
            'TrainedF1': ['mean', 'std'],
            'TotalRuntime': ['mean', 'std']
        }).round(4)
        drop_strategy_summary.columns = ['_'.join(col).strip() for col in drop_strategy_summary.columns]
        drop_strategy_summary.to_csv(TABLES_DIR / "drop_strategy_comparison.csv")
        
        # Dataset vulnerability ranking
        dataset_vulnerability = self.df.groupby('Dataset').agg({
            'ReidentificationRatePercent': ['mean', 'max', 'std'],
            'TrainedF1': 'mean',
            'DatasetSize': 'first'
        }).round(4)
        dataset_vulnerability.columns = ['_'.join(col).strip() for col in dataset_vulnerability.columns]
        dataset_vulnerability = dataset_vulnerability.sort_values('ReidentificationRatePercent_mean', ascending=False)
        dataset_vulnerability.to_csv(TABLES_DIR / "dataset_vulnerability_ranking.csv")
        
        print(f"📊 Statistical summary tables saved to {TABLES_DIR}")
    
    def create_detailed_encoding_profiles(self):
        """Create detailed analysis profiles for each encoding scheme."""
        print("📝 Creating detailed encoding scheme profiles...")
        
        for encoding in self.df['Encoding'].unique():
            encoding_data = self.df[self.df['Encoding'] == encoding]
            
            # Create encoding-specific analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Plot 1: Performance metrics distribution
            ax = axes[0]
            metrics = ['TrainedPrecision', 'TrainedRecall', 'TrainedF1']
            encoding_data[metrics].boxplot(ax=ax)
            ax.set_title(f'{encoding}: Performance Metrics Distribution')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Re-identification rates by overlap
            ax = axes[1]
            for drop_strategy in encoding_data['DropFrom'].unique():
                strategy_data = encoding_data[encoding_data['DropFrom'] == drop_strategy]
                overlap_grouped = strategy_data.groupby('Overlap')['ReidentificationRatePercent'].mean()
                ax.plot(overlap_grouped.index, overlap_grouped.values, 
                       marker='o', label=f'Drop: {drop_strategy}', linewidth=2)
            ax.set_title(f'{encoding}: Re-identification Rate vs Overlap')
            ax.set_xlabel('Overlap')
            ax.set_ylabel('Re-identification Rate (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Runtime breakdown
            ax = axes[2]
            runtime_cols = ['GraphMatchingAttackTime', 'HyperparameterOptimizationTime', 
                          'ModelTrainingTime', 'ApplicationtoEncodedDataTime', 
                          'RefinementandReconstructionTime']
            runtime_data = encoding_data[runtime_cols].mean()
            colors = plt.cm.Set3(np.linspace(0, 1, len(runtime_cols)))
            bars = ax.bar(range(len(runtime_data)), runtime_data.values, color=colors)
            ax.set_title(f'{encoding}: Runtime Breakdown')
            ax.set_ylabel('Time (minutes)')
            ax.set_xticks(range(len(runtime_data)))
            ax.set_xticklabels(['Graph\nMatching', 'Hyperparameter\nOptimization', 
                              'Model\nTraining', 'Application\nto Data', 
                              'Refinement\n& Reconstruction'], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, val in zip(bars, runtime_data.values):
                ax.text(bar.get_x() + bar.get_width()/2., val + val*0.01,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 4: Dataset size impact
            ax = axes[3]
            size_data = encoding_data.dropna(subset=['DatasetSize'])
            if len(size_data) > 0:
                ax.scatter(size_data['DatasetSize'], size_data['ReidentificationRatePercent'], 
                          alpha=0.7, s=50)
                ax.set_title(f'{encoding}: Dataset Size Impact')
                ax.set_xlabel('Dataset Size')
                ax.set_ylabel('Re-identification Rate (%)')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No dataset size data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{encoding}: Dataset Size Impact')
            
            # Plot 5: Hyperparameter correlation
            ax = axes[4]
            if 'HypOpF1' in encoding_data.columns and not encoding_data['HypOpF1'].isna().all():
                ax.scatter(encoding_data['HypOpF1'], encoding_data['TrainedF1'], alpha=0.7, s=50)
                # Add diagonal line
                min_val = min(encoding_data['HypOpF1'].min(), encoding_data['TrainedF1'].min())
                max_val = max(encoding_data['HypOpF1'].max(), encoding_data['TrainedF1'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                ax.set_title(f'{encoding}: Validation vs Training F1')
                ax.set_xlabel('Hyperparameter Optimization F1')
                ax.set_ylabel('Trained F1')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No hyperparameter F1 data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{encoding}: Validation vs Training F1')
            
            # Plot 6: Summary statistics table
            ax = axes[5]
            ax.axis('off')
            
            # Create summary stats text
            stats_text = f"""
            {encoding} Summary Statistics:
            
            Experiments: {len(encoding_data)}
            
            Re-identification Rate:
              Mean: {encoding_data['ReidentificationRatePercent'].mean():.2f}%
              Max: {encoding_data['ReidentificationRatePercent'].max():.2f}%
              Std: {encoding_data['ReidentificationRatePercent'].std():.2f}%
            
            F1 Score:
              Mean: {encoding_data['TrainedF1'].mean():.3f}
              Max: {encoding_data['TrainedF1'].max():.3f}
              Std: {encoding_data['TrainedF1'].std():.3f}
            
            Total Runtime:
              Mean: {encoding_data['TotalRuntime'].mean():.1f} min
              Max: {encoding_data['TotalRuntime'].max():.1f} min
            """
            
            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle(f'Detailed Analysis Profile: {encoding} Encoding Scheme', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"detailed_profile_{encoding.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive DEA analysis...")
        print("=" * 60)
        
        # Generate executive summary
        summary = self.generate_executive_summary()
        
        # Create all visualizations
        self.plot_encoding_comparison_comprehensive()
        self.plot_overlap_impact_analysis()
        self.plot_dataset_scalability_analysis()
        self.plot_privacy_risk_heatmap()
        self.plot_performance_vs_privacy_tradeoff()
        
        # Generate detailed profiles
        self.create_detailed_encoding_profiles()
        
        # Generate statistical tables
        self.generate_statistical_summary_tables()
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete! Results saved to:")
        print(f"   📊 Plots: {PLOTS_DIR}")
        print(f"   📋 Tables: {TABLES_DIR}")
        print(f"   📈 Summaries: {SUMMARY_DIR}")
        print("\n🎯 Key Findings:")
        print(f"   • Average re-identification rate: {summary['avg_reidentification_rate']:.2f}%")
        print(f"   • Best performing encoding: {max(summary['encoding_performance'], key=lambda x: summary['encoding_performance'][x]['avg_f1_score'])}")
        print(f"   • Most vulnerable encoding: {max(summary['encoding_performance'], key=lambda x: summary['encoding_performance'][x]['avg_reidentification_rate'])}")


if __name__ == "__main__":
    # Initialize and run the analysis
    analyzer = DEAAnalyzer()
    analyzer.run_complete_analysis()