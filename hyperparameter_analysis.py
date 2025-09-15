#!/usr/bin/env python3
"""
Hyperparameter Analysis Script for Dataset Extension Attack Experiments

This script analyzes hyperparameter combinations for each encoding scheme:
- BloomFilter
- TabMinHash  
- TwoStepHash

It identifies the most chosen and best performing hyperparameter combinations,
with bucketing/averaging for continuous values.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HyperparameterAnalyzer:
    def __init__(self, experiment_results_dir="experiment_results"):
        self.experiment_results_dir = Path(experiment_results_dir)
        self.encoding_schemes = ["BloomFilter", "TabMinHash", "TwoStepHash"]
        self.data = defaultdict(list)
        self.performance_metrics = ["average_dice", "average_precision", "average_recall", "average_f1"]
        
    def load_experiment_data(self):
        """Load all experiment data from the results directory."""
        print("Loading experiment data...")
        
        for experiment_dir in self.experiment_results_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            # Extract encoding scheme from directory name
            dir_name = experiment_dir.name
            encoding_scheme = None
            for scheme in self.encoding_schemes:
                if scheme in dir_name:
                    encoding_scheme = scheme
                    break
            
            if encoding_scheme is None:
                continue
                
            # Load config and best result
            config_path = experiment_dir / "config.json"
            best_result_path = experiment_dir / "hyperparameteroptimization" / "best_result.json"
            
            if not config_path.exists() or not best_result_path.exists():
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                with open(best_result_path, 'r') as f:
                    best_result = json.load(f)
                    
                # Extract relevant information
                experiment_data = {
                    'encoding_scheme': encoding_scheme,
                    'experiment_dir': str(experiment_dir),
                    'dataset': self._extract_dataset_name(dir_name),
                    'config': config,
                    'hyperparameters': best_result,
                    'performance': {metric: best_result.get(metric, 0) for metric in self.performance_metrics}
                }
                
                self.data[encoding_scheme].append(experiment_data)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {experiment_dir}: {e}")
                continue
                
        print(f"Loaded {sum(len(experiments) for experiments in self.data.values())} experiments")
        for scheme, experiments in self.data.items():
            print(f"  {scheme}: {len(experiments)} experiments")
    
    def _extract_dataset_name(self, dir_name):
        """Extract dataset name from experiment directory name."""
        # Remove experiment prefix and timestamp
        parts = dir_name.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[2:-1])  # Remove first two parts and last timestamp part
        return "unknown"
    
    def analyze_hyperparameters(self):
        """Analyze hyperparameters for each encoding scheme."""
        results = {}
        
        for encoding_scheme in self.encoding_schemes:
            if encoding_scheme not in self.data:
                continue
                
            print(f"\n=== Analyzing {encoding_scheme} ===")
            experiments = self.data[encoding_scheme]
            
            # Extract hyperparameters
            hyperparams = self._extract_hyperparameters(experiments)
            
            # Analyze most chosen combinations
            most_chosen = self._find_most_chosen_combinations(hyperparams)
            
            # Analyze best performing combinations
            best_performing = self._find_best_performing_combinations(hyperparams)
            
            # Analyze individual parameter distributions
            param_distributions = self._analyze_parameter_distributions(hyperparams)
            
            # Find recommended configuration
            recommended_config = self._find_recommended_configuration(hyperparams)
            
            results[encoding_scheme] = {
                'most_chosen': most_chosen,
                'best_performing': best_performing,
                'param_distributions': param_distributions,
                'recommended_config': recommended_config,
                'total_experiments': len(experiments)
            }
            
        return results
    
    def _extract_hyperparameters(self, experiments):
        """Extract hyperparameters from experiments."""
        hyperparams = []
        
        for exp in experiments:
            hp = exp['hyperparameters'].copy()
            hp['performance'] = exp['performance']
            hp['dataset'] = exp['dataset']
            hyperparams.append(hp)
            
        return hyperparams
    
    def _find_most_chosen_combinations(self, hyperparams):
        """Find the most frequently chosen hyperparameter combinations."""
        # Create combinations of key hyperparameters with better clustering
        combinations = []
        
        for hp in hyperparams:
            # Focus on the most important hyperparameters for clustering
            key_params = {
                'activation_fn': hp.get('activation_fn'),
                'optimizer_name': hp.get('optimizer', {}).get('name'),
                'loss_fn': hp.get('loss_fn'),
                'batch_size': hp.get('batch_size'),
                'lr_scheduler_name': hp.get('lr_scheduler', {}).get('name'),
                'hidden_layer_size': self._bucket_hidden_layer_size(hp.get('hidden_layer_size')),
                'dropout_rate': self._bucket_dropout_rate(hp.get('dropout_rate')),
                'optimizer_lr': self._bucket_learning_rate(hp.get('optimizer', {}).get('lr')),
                'threshold': self._bucket_threshold(hp.get('threshold'))
            }
            
            # Create a string representation of the combination
            combo_str = " | ".join([f"{k}={v}" for k, v in key_params.items() if v is not None])
            combinations.append((combo_str, key_params, hp['performance']))
        
        # Count combinations
        combo_counts = Counter([combo[0] for combo in combinations])
        
        # Get top combinations
        top_combinations = []
        for combo_str, count in combo_counts.most_common(10):
            # Find examples of this combination
            examples = [combo for combo in combinations if combo[0] == combo_str]
            avg_performance = self._calculate_average_performance([ex[2] for ex in examples])
            
            top_combinations.append({
                'combination': combo_str,
                'count': count,
                'percentage': (count / len(hyperparams)) * 100,
                'avg_performance': avg_performance,
                'examples': examples[:3]  # Show first 3 examples
            })
        
        return top_combinations
    
    def _find_best_performing_combinations(self, hyperparams):
        """Find the best performing hyperparameter combinations."""
        # Sort by average_dice (primary metric)
        sorted_hp = sorted(hyperparams, key=lambda x: x['performance']['average_dice'], reverse=True)
        
        # Get top 10 performing combinations
        top_performing = []
        for i, hp in enumerate(sorted_hp[:10]):
            # Create combination string
            key_params = {
                'num_layers': hp.get('num_layers'),
                'hidden_layer_size': hp.get('hidden_layer_size'),
                'dropout_rate': hp.get('dropout_rate'),
                'activation_fn': hp.get('activation_fn'),
                'optimizer_name': hp.get('optimizer', {}).get('name'),
                'optimizer_lr': hp.get('optimizer', {}).get('lr'),
                'loss_fn': hp.get('loss_fn'),
                'threshold': hp.get('threshold'),
                'batch_size': hp.get('batch_size'),
                'lr_scheduler_name': hp.get('lr_scheduler', {}).get('name'),
                'lr_scheduler_mode': hp.get('lr_scheduler', {}).get('mode_cyclic')
            }
            
            combo_str = " | ".join([f"{k}={v}" for k, v in key_params.items() if v is not None])
            
            top_performing.append({
                'rank': i + 1,
                'combination': combo_str,
                'performance': hp['performance'],
                'dataset': hp['dataset']
            })
        
        return top_performing
    
    def _find_recommended_configuration(self, hyperparams):
        """Find the recommended configuration based on top performers and common patterns."""
        # Get top 20% of experiments by performance
        sorted_hp = sorted(hyperparams, key=lambda x: x['performance']['average_dice'], reverse=True)
        top_performers = sorted_hp[:max(1, len(sorted_hp) // 5)]  # Top 20%
        
        # Analyze patterns in top performers
        param_analysis = {}
        
        # Key parameters to analyze
        key_params = [
            'activation_fn', 'optimizer_name', 'loss_fn', 'batch_size', 
            'lr_scheduler_name', 'num_layers'
        ]
        
        for param in key_params:
            values = []
            for hp in top_performers:
                if param.startswith('optimizer_'):
                    value = hp.get('optimizer', {}).get(param.split('_', 1)[1])
                elif param.startswith('lr_scheduler_'):
                    value = hp.get('lr_scheduler', {}).get(param.split('_', 2)[2])
                else:
                    value = hp.get(param)
                
                if value is not None:
                    values.append(value)
            
            if values:
                # Find most common value in top performers
                value_counts = Counter(values)
                most_common = value_counts.most_common(1)[0]
                
                param_analysis[param] = {
                    'recommended': most_common[0],
                    'frequency': most_common[1],
                    'percentage': (most_common[1] / len(values)) * 100,
                    'all_values': dict(value_counts)
                }
        
        # For continuous parameters, find the median/mean of top performers
        continuous_params = ['dropout_rate', 'threshold']
        for param in continuous_params:
            values = [hp.get(param) for hp in top_performers if hp.get(param) is not None]
            if values:
                param_analysis[param] = {
                    'recommended': np.median(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Learning rate analysis
        lr_values = [hp.get('optimizer', {}).get('lr') for hp in top_performers 
                    if hp.get('optimizer', {}).get('lr') is not None]
        if lr_values:
            param_analysis['optimizer_lr'] = {
                'recommended': np.median(lr_values),
                'mean': np.mean(lr_values),
                'std': np.std(lr_values),
                'min': np.min(lr_values),
                'max': np.max(lr_values),
                'count': len(lr_values)
            }
        
        # Hidden layer size analysis
        hidden_sizes = [hp.get('hidden_layer_size') for hp in top_performers 
                       if hp.get('hidden_layer_size') is not None]
        if hidden_sizes:
            param_analysis['hidden_layer_size'] = {
                'recommended': int(np.median(hidden_sizes)),
                'mean': np.mean(hidden_sizes),
                'std': np.std(hidden_sizes),
                'min': np.min(hidden_sizes),
                'max': np.max(hidden_sizes),
                'count': len(hidden_sizes)
            }
        
        return param_analysis
    
    def _analyze_parameter_distributions(self, hyperparams):
        """Analyze distributions of individual parameters."""
        distributions = {}
        
        # Define parameters to analyze
        param_configs = {
            'num_layers': {'type': 'categorical'},
            'hidden_layer_size': {'type': 'categorical'},
            'dropout_rate': {'type': 'continuous', 'buckets': [0.1, 0.2, 0.3, 0.4, 0.5]},
            'activation_fn': {'type': 'categorical'},
            'optimizer_name': {'type': 'categorical'},
            'optimizer_lr': {'type': 'continuous', 'buckets': [1e-5, 1e-4, 1e-3, 1e-2]},
            'loss_fn': {'type': 'categorical'},
            'threshold': {'type': 'continuous', 'buckets': [0.1, 0.2, 0.3, 0.4, 0.5]},
            'batch_size': {'type': 'categorical'},
            'lr_scheduler_name': {'type': 'categorical'},
            'lr_scheduler_mode': {'type': 'categorical'}
        }
        
        for param, config in param_configs.items():
            values = []
            performances = []
            
            for hp in hyperparams:
                if param.startswith('optimizer_'):
                    value = hp.get('optimizer', {}).get(param.split('_', 1)[1])
                elif param.startswith('lr_scheduler_'):
                    value = hp.get('lr_scheduler', {}).get(param.split('_', 2)[2])
                else:
                    value = hp.get(param)
                
                if value is not None:
                    # Convert to string to make it hashable
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    elif config['type'] == 'continuous':
                        value = self._bucket_continuous_value(value, config['buckets'])
                    
                    values.append(value)
                    performances.append(hp['performance']['average_dice'])
            
            if values:
                # Count occurrences
                value_counts = Counter(values)
                
                # Calculate average performance for each value
                value_performance = defaultdict(list)
                for val, perf in zip(values, performances):
                    value_performance[val].append(perf)
                
                avg_performance = {val: np.mean(perfs) for val, perfs in value_performance.items()}
                
                distributions[param] = {
                    'counts': dict(value_counts),
                    'percentages': {val: (count / len(values)) * 100 for val, count in value_counts.items()},
                    'avg_performance': avg_performance,
                    'total_samples': len(values)
                }
        
        return distributions
    
    def _bucket_continuous_value(self, value, buckets):
        """Bucket a continuous value into discrete categories."""
        if value is None:
            return None
        
        for i, bucket in enumerate(buckets):
            if value <= bucket:
                return f"<={bucket}"
        
        return f">{buckets[-1]}"
    
    def _bucket_hidden_layer_size(self, value):
        """Bucket hidden layer size into meaningful categories."""
        if value is None:
            return None
        
        if value <= 1024:
            return "small (<=1024)"
        elif value <= 2048:
            return "medium (1025-2048)"
        else:
            return "large (>2048)"
    
    def _bucket_dropout_rate(self, value):
        """Bucket dropout rate into meaningful categories."""
        if value is None:
            return None
        
        if value <= 0.15:
            return "low (<=0.15)"
        elif value <= 0.25:
            return "medium (0.15-0.25)"
        elif value <= 0.35:
            return "high (0.25-0.35)"
        else:
            return "very_high (>0.35)"
    
    def _bucket_learning_rate(self, value):
        """Bucket learning rate into meaningful categories."""
        if value is None:
            return None
        
        if value <= 1e-5:
            return "very_low (<=1e-5)"
        elif value <= 1e-4:
            return "low (1e-5 to 1e-4)"
        elif value <= 1e-3:
            return "medium (1e-4 to 1e-3)"
        else:
            return "high (>1e-3)"
    
    def _bucket_threshold(self, value):
        """Bucket threshold into meaningful categories."""
        if value is None:
            return None
        
        if value <= 0.2:
            return "low (<=0.2)"
        elif value <= 0.3:
            return "medium (0.2-0.3)"
        elif value <= 0.4:
            return "high (0.3-0.4)"
        else:
            return "very_high (>0.4)"
    
    def _calculate_average_performance(self, performances):
        """Calculate average performance across multiple experiments."""
        if not performances:
            return {}
        
        avg_perf = {}
        for metric in self.performance_metrics:
            values = [perf.get(metric, 0) for perf in performances if perf.get(metric) is not None]
            if values:
                avg_perf[metric] = np.mean(values)
        
        return avg_perf
    
    def generate_report(self, results):
        """Generate a comprehensive analysis report."""
        report = []
        report.append("# Hyperparameter Analysis Report")
        report.append("=" * 50)
        
        for encoding_scheme, data in results.items():
            report.append(f"\n## {encoding_scheme} Analysis")
            report.append(f"Total Experiments: {data['total_experiments']}")
            report.append("-" * 30)
            
            # Most chosen combinations
            report.append("\n### Most Chosen Hyperparameter Combinations")
            for i, combo in enumerate(data['most_chosen'][:5], 1):
                report.append(f"\n{i}. **Count: {combo['count']} ({combo['percentage']:.1f}%)**")
                report.append(f"   Combination: {combo['combination']}")
                if combo['avg_performance']:
                    report.append(f"   Avg Dice: {combo['avg_performance'].get('average_dice', 0):.4f}")
            
            # Best performing combinations
            report.append("\n### Best Performing Hyperparameter Combinations")
            for combo in data['best_performing'][:5]:
                report.append(f"\n{combo['rank']}. **Dice: {combo['performance']['average_dice']:.4f}**")
                report.append(f"   Dataset: {combo['dataset']}")
                report.append(f"   Combination: {combo['combination']}")
            
            # Recommended configuration
            report.append("\n### Recommended Configuration (Based on Top 20% Performers)")
            if 'recommended_config' in data and data['recommended_config']:
                for param, config in data['recommended_config'].items():
                    if 'recommended' in config:
                        if isinstance(config['recommended'], (int, float)):
                            if 'frequency' in config:
                                report.append(f"  - **{param}**: {config['recommended']} (used in {config['frequency']}/{config.get('count', 'N/A')} top experiments, {config['percentage']:.1f}%)")
                            else:
                                report.append(f"  - **{param}**: {config['recommended']:.4f} (median of top performers, range: {config['min']:.4f}-{config['max']:.4f})")
                        else:
                            if 'frequency' in config:
                                report.append(f"  - **{param}**: {config['recommended']} (used in {config['frequency']}/{config.get('count', 'N/A')} top experiments, {config['percentage']:.1f}%)")
                            else:
                                report.append(f"  - **{param}**: {config['recommended']}")
            
            # Parameter distributions
            report.append("\n### Parameter Distributions")
            for param, dist in data['param_distributions'].items():
                report.append(f"\n**{param}:**")
                for value, count in sorted(dist['counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    percentage = dist['percentages'][value]
                    avg_perf = dist['avg_performance'].get(value, 0)
                    report.append(f"  - {value}: {count} ({percentage:.1f}%) - Avg Dice: {avg_perf:.4f}")
        
        return "\n".join(report)
    
    def save_results(self, results, output_dir="analysis"):
        """Save analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        with open(output_dir / "hyperparameter_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save report as text
        report = self.generate_report(results)
        with open(output_dir / "hyperparameter_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Create visualizations
        self._create_visualizations(results, output_dir)
        
        print(f"\nResults saved to {output_dir}/")
    
    def _create_visualizations(self, results, output_dir):
        """Create visualizations for the analysis."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with subplots for each encoding scheme
        fig, axes = plt.subplots(len(self.encoding_schemes), 2, figsize=(15, 5 * len(self.encoding_schemes)))
        if len(self.encoding_schemes) == 1:
            axes = axes.reshape(1, -1)
        
        for i, encoding_scheme in enumerate(self.encoding_schemes):
            if encoding_scheme not in results:
                continue
                
            data = results[encoding_scheme]
            
            # Plot 1: Most chosen combinations
            ax1 = axes[i, 0]
            most_chosen = data['most_chosen'][:5]
            if most_chosen:
                counts = [combo['count'] for combo in most_chosen]
                labels = [f"Combo {j+1}" for j in range(len(most_chosen))]
                ax1.bar(labels, counts)
                ax1.set_title(f'{encoding_scheme} - Most Chosen Combinations')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Best performing combinations
            ax2 = axes[i, 1]
            best_performing = data['best_performing'][:5]
            if best_performing:
                dice_scores = [combo['performance']['average_dice'] for combo in best_performing]
                labels = [f"Rank {combo['rank']}" for combo in best_performing]
                ax2.bar(labels, dice_scores)
                ax2.set_title(f'{encoding_scheme} - Best Performing Combinations')
                ax2.set_ylabel('Average Dice Score')
                ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "hyperparameter_analysis_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed parameter distribution plots
        for encoding_scheme in self.encoding_schemes:
            if encoding_scheme not in results:
                continue
                
            data = results[encoding_scheme]
            param_dist = data['param_distributions']
            
            # Select top parameters to visualize
            top_params = ['activation_fn', 'optimizer_name', 'loss_fn', 'batch_size']
            available_params = [p for p in top_params if p in param_dist]
            
            if available_params:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                for i, param in enumerate(available_params[:4]):
                    if i >= len(axes):
                        break
                        
                    dist = param_dist[param]
                    values = list(dist['counts'].keys())
                    counts = list(dist['counts'].values())
                    
                    axes[i].bar(values, counts)
                    axes[i].set_title(f'{encoding_scheme} - {param}')
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
                
                # Hide unused subplots
                for i in range(len(available_params), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{encoding_scheme.lower()}_parameter_distributions.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()

def main():
    """Main function to run the hyperparameter analysis."""
    print("Starting Hyperparameter Analysis...")
    
    # Initialize analyzer
    analyzer = HyperparameterAnalyzer()
    
    # Load data
    analyzer.load_experiment_data()
    
    # Analyze hyperparameters
    results = analyzer.analyze_hyperparameters()
    
    # Save results
    analyzer.save_results(results)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for encoding_scheme, data in results.items():
        print(f"\n{encoding_scheme}:")
        print(f"  Total experiments: {data['total_experiments']}")
        
        if data['most_chosen']:
            top_combo = data['most_chosen'][0]
            print(f"  Most chosen combination: {top_combo['count']} times ({top_combo['percentage']:.1f}%)")
        
        if data['best_performing']:
            best_combo = data['best_performing'][0]
            print(f"  Best performance: Dice = {best_combo['performance']['average_dice']:.4f}")
        
        if 'recommended_config' in data and data['recommended_config']:
            print(f"  Recommended config:")
            for param, config in data['recommended_config'].items():
                if 'recommended' in config:
                    if isinstance(config['recommended'], (int, float)):
                        print(f"    {param}: {config['recommended']:.4f}")
                    else:
                        print(f"    {param}: {config['recommended']}")
    
    print(f"\nDetailed results saved to analysis/ directory")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
