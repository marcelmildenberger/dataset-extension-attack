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
import numpy as np
from collections import defaultdict, Counter
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
        """Find the recommended configuration based on weighted top performers."""
        # Sort by performance and get top 30% for weighted analysis
        sorted_hp = sorted(hyperparams, key=lambda x: x['performance']['average_dice'], reverse=True)
        top_performers = sorted_hp[:max(1, int(len(sorted_hp) * 0.3))]  # Top 30%
        
        # Calculate weights based on performance (higher performance = higher weight)
        max_dice = max(hp['performance']['average_dice'] for hp in top_performers)
        min_dice = min(hp['performance']['average_dice'] for hp in top_performers)
        
        # Normalize weights so they sum to 1
        weights = []
        for hp in top_performers:
            # Linear weighting based on performance
            weight = (hp['performance']['average_dice'] - min_dice) / (max_dice - min_dice) if max_dice > min_dice else 1.0
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        param_analysis = {}
        
        # Key parameters to analyze
        key_params = [
            'activation_fn', 'optimizer_name', 'loss_fn', 'batch_size', 
            'lr_scheduler_name', 'num_layers'
        ]
        
        for param in key_params:
            value_weights = defaultdict(float)
            
            for hp, weight in zip(top_performers, weights):
                if param.startswith('optimizer_'):
                    value = hp.get('optimizer', {}).get(param.split('_', 1)[1])
                elif param.startswith('lr_scheduler_'):
                    value = hp.get('lr_scheduler', {}).get(param.split('_', 2)[2])
                else:
                    value = hp.get(param)
                
                if value is not None:
                    value_weights[value] += weight
            
            if value_weights:
                # Find the value with highest weighted frequency
                best_value = max(value_weights.items(), key=lambda x: x[1])
                
                param_analysis[param] = {
                    'recommended': best_value[0],
                    'weighted_score': best_value[1],
                    'all_values': dict(value_weights)
                }
        
        # For continuous parameters, calculate weighted median/mean
        continuous_params = ['dropout_rate', 'threshold']
        for param in continuous_params:
            weighted_values = []
            for hp, weight in zip(top_performers, weights):
                value = hp.get(param)
                if value is not None:
                    # Add the value multiple times based on its weight
                    weighted_values.extend([value] * int(weight * 1000))  # Scale up for better precision
            
            if weighted_values:
                param_analysis[param] = {
                    'recommended': np.median(weighted_values),
                    'weighted_mean': np.mean(weighted_values),
                    'std': np.std(weighted_values),
                    'min': np.min(weighted_values),
                    'max': np.max(weighted_values),
                    'count': len(weighted_values)
                }
        
        # Learning rate analysis with weights
        weighted_lr_values = []
        for hp, weight in zip(top_performers, weights):
            lr = hp.get('optimizer', {}).get('lr')
            if lr is not None:
                weighted_lr_values.extend([lr] * int(weight * 1000))
        
        if weighted_lr_values:
            param_analysis['optimizer_lr'] = {
                'recommended': np.median(weighted_lr_values),
                'weighted_mean': np.mean(weighted_lr_values),
                'std': np.std(weighted_lr_values),
                'min': np.min(weighted_lr_values),
                'max': np.max(weighted_lr_values),
                'count': len(weighted_lr_values)
            }
        
        # Hidden layer size analysis with weights
        weighted_hidden_sizes = []
        for hp, weight in zip(top_performers, weights):
            size = hp.get('hidden_layer_size')
            if size is not None:
                weighted_hidden_sizes.extend([size] * int(weight * 1000))
        
        if weighted_hidden_sizes:
            param_analysis['hidden_layer_size'] = {
                'recommended': int(np.median(weighted_hidden_sizes)),
                'weighted_mean': np.mean(weighted_hidden_sizes),
                'std': np.std(weighted_hidden_sizes),
                'min': np.min(weighted_hidden_sizes),
                'max': np.max(weighted_hidden_sizes),
                'count': len(weighted_hidden_sizes)
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
        """Generate a focused report with only weighted recommended configurations."""
        report = []
        report.append("# Hyperparameter Analysis Report - Weighted Recommendations")
        report.append("=" * 70)
        report.append("Based on top 30% performers with performance-weighted analysis")
        report.append("=" * 70)
        
        for encoding_scheme, data in results.items():
            report.append(f"\n## {encoding_scheme} - Recommended Configuration")
            report.append(f"Total Experiments Analyzed: {data['total_experiments']}")
            report.append("-" * 50)
            
            # Recommended configuration with weighted analysis
            if 'recommended_config' in data and data['recommended_config']:
                report.append("\n### Weighted Recommended Hyperparameters:")
                report.append("(Higher performing configurations have greater influence)")
                report.append("")
                
                for param, config in data['recommended_config'].items():
                    if 'recommended' in config:
                        if isinstance(config['recommended'], (int, float)):
                            if 'weighted_score' in config:
                                report.append(f"  • **{param}**: {config['recommended']:.4f} (weighted score: {config['weighted_score']:.3f})")
                            else:
                                report.append(f"  • **{param}**: {config['recommended']:.4f} (weighted median, range: {config['min']:.4f}-{config['max']:.4f})")
                        else:
                            if 'weighted_score' in config:
                                report.append(f"  • **{param}**: {config['recommended']} (weighted score: {config['weighted_score']:.3f})")
                            else:
                                report.append(f"  • **{param}**: {config['recommended']}")
                
                report.append("")
                report.append("### Weighting Methodology:")
                report.append("- Top 30% of experiments by average_dice score are analyzed")
                report.append("- Each configuration is weighted by its performance relative to others")
                report.append("- Higher performing configurations have proportionally greater influence")
                report.append("- Categorical parameters: most frequent weighted choice")
                report.append("- Continuous parameters: weighted median/mean")
            else:
                report.append("\nNo recommended configuration available (insufficient data)")
        
        return "\n".join(report)
    
    def save_results(self, results, output_dir="analysis"):
        """Save only the text report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save report as text
        report = self.generate_report(results)
        with open(output_dir / "hyperparameter_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport saved to {output_dir}/hyperparameter_analysis_report.txt")
    

def main():
    """Main function to run the hyperparameter analysis."""
    print("Starting Weighted Hyperparameter Analysis...")
    
    # Initialize analyzer
    analyzer = HyperparameterAnalyzer()
    
    # Load data
    analyzer.load_experiment_data()
    
    # Analyze hyperparameters
    results = analyzer.analyze_hyperparameters()
    
    # Save results
    analyzer.save_results(results)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()