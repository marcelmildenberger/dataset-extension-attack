#!/usr/bin/env python3
"""
Hyperparameter Analysis Script for Dataset Extension Attack Experiments

This script analyzes hyperparameter combinations for each encoding scheme:
- BloomFilter
- TabMinHash  
- TwoStepHash

It identifies the most chosen and best performing hyperparameter combinations,
with bucketing/averaging for continuous values.

Additionally, it generates narrow Ray Tune search spaces based on the top 15%
performing experiments, providing optimized parameter ranges for future
hyperparameter optimization runs.

Outputs:
- Detailed analysis report with recommended configurations
- Optimal configurations JSON for direct use
- Narrow search spaces JSON with Ray Tune code for each encoding scheme
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
            
        # Generate narrow search spaces after all analysis is complete
        search_spaces = self._generate_narrow_search_spaces(results)
        
        # Add search spaces to results
        for encoding_scheme, search_data in search_spaces.items():
            if encoding_scheme in results:
                results[encoding_scheme]['search_spaces'] = search_data
            
        return results
    
    def _extract_hyperparameters(self, experiments):
        """Extract hyperparameters from experiments."""
        hyperparams = []
        
        for exp in experiments:
            hp = exp['hyperparameters'].copy()
            hp['performance'] = exp['performance']
            hp['dataset'] = exp['dataset']
            
            # Flatten optimizer and scheduler parameters for easier analysis
            if 'optimizer' in hp and isinstance(hp['optimizer'], dict):
                optimizer = hp['optimizer']
                hp['optimizer_name'] = optimizer.get('name')
                hp['optimizer_lr'] = optimizer.get('lr')
                hp['optimizer_momentum'] = optimizer.get('momentum', 0.0)
                hp['optimizer_weight_decay'] = optimizer.get('weight_decay', 0.0)
                hp['optimizer_betas'] = optimizer.get('betas', (0.9, 0.999))
                hp['optimizer_eps'] = optimizer.get('eps', 1e-8)
                hp['optimizer_alpha'] = optimizer.get('alpha', 0.99)
                
            if 'lr_scheduler' in hp and isinstance(hp['lr_scheduler'], dict):
                scheduler = hp['lr_scheduler']
                hp['lr_scheduler_name'] = scheduler.get('name')
                hp['lr_scheduler_mode'] = scheduler.get('mode', 'min')
                hp['lr_scheduler_factor'] = scheduler.get('factor', 0.1)
                hp['lr_scheduler_patience'] = scheduler.get('patience', 10)
                hp['lr_scheduler_T_max'] = scheduler.get('T_max', 10)
                hp['lr_scheduler_eta_min'] = scheduler.get('eta_min', 0)
                hp['lr_scheduler_base_lr'] = scheduler.get('base_lr', 1e-5)
                hp['lr_scheduler_max_lr'] = scheduler.get('max_lr', 1e-1)
                hp['lr_scheduler_step_size_up'] = scheduler.get('step_size_up', 2000)
                hp['lr_scheduler_mode_cyclic'] = scheduler.get('mode_cyclic', 'triangular')
                hp['lr_scheduler_gamma'] = scheduler.get('gamma', 0.1)
                hp['lr_scheduler_step_size'] = scheduler.get('step_size', 30)
                
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
                'optimizer_name': hp.get('optimizer_name'),
                'loss_fn': hp.get('loss_fn'),
                'batch_size': hp.get('batch_size'),
                'lr_scheduler_name': hp.get('lr_scheduler_name'),
                'hidden_layer_size': self._bucket_hidden_layer_size(hp.get('hidden_layer_size')),
                'dropout_rate': self._bucket_dropout_rate(hp.get('dropout_rate')),
                'optimizer_lr': self._bucket_learning_rate(hp.get('optimizer_lr')),
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
            # Create combination string with detailed parameters
            key_params = {
                'num_layers': hp.get('num_layers'),
                'hidden_layer_size': hp.get('hidden_layer_size'),
                'dropout_rate': hp.get('dropout_rate'),
                'activation_fn': hp.get('activation_fn'),
                'optimizer_name': hp.get('optimizer_name'),
                'optimizer_lr': hp.get('optimizer_lr'),
                'optimizer_momentum': hp.get('optimizer_momentum'),
                'optimizer_weight_decay': hp.get('optimizer_weight_decay'),
                'loss_fn': hp.get('loss_fn'),
                'threshold': hp.get('threshold'),
                'batch_size': hp.get('batch_size'),
                'lr_scheduler_name': hp.get('lr_scheduler_name'),
                'lr_scheduler_mode': hp.get('lr_scheduler_mode'),
                'lr_scheduler_factor': hp.get('lr_scheduler_factor'),
                'lr_scheduler_patience': hp.get('lr_scheduler_patience'),
                'lr_scheduler_T_max': hp.get('lr_scheduler_T_max'),
                'lr_scheduler_eta_min': hp.get('lr_scheduler_eta_min')
            }
            
            combo_str = " | ".join([f"{k}={v}" for k, v in key_params.items() if v is not None])
            
            top_performing.append({
                'rank': i + 1,
                'combination': combo_str,
                'performance': hp['performance'],
                'dataset': hp['dataset'],
                'full_config': hp  # Include full config for detailed analysis
            })
        
        return top_performing
    
    def _find_recommended_configuration(self, hyperparams):
        """Find the recommended configuration based on weighted top performers."""
        # Sort by performance and get top 15% for weighted analysis
        sorted_hp = sorted(hyperparams, key=lambda x: x['performance']['average_dice'], reverse=True)
        top_performers = sorted_hp[:max(1, int(len(sorted_hp) * 0.15))]  # Top 15%
        
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
        
        # Key parameters to analyze (categorical)
        categorical_params = [
            'activation_fn', 'optimizer_name', 'loss_fn', 'batch_size', 
            'lr_scheduler_name', 'num_layers', 'lr_scheduler_mode', 'lr_scheduler_mode_cyclic'
        ]
        
        for param in categorical_params:
            value_weights = defaultdict(float)
            
            for hp, weight in zip(top_performers, weights):
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
        continuous_params = [
            'dropout_rate', 'threshold', 'optimizer_lr', 'optimizer_momentum', 
            'optimizer_weight_decay', 'optimizer_eps', 'optimizer_alpha',
            'lr_scheduler_factor', 'lr_scheduler_patience', 'lr_scheduler_T_max',
            'lr_scheduler_eta_min', 'lr_scheduler_base_lr', 'lr_scheduler_max_lr',
            'lr_scheduler_step_size_up', 'lr_scheduler_gamma', 'lr_scheduler_step_size',
            'hidden_layer_size'
        ]
        
        for param in continuous_params:
            weighted_values = []
            for hp, weight in zip(top_performers, weights):
                value = hp.get(param)
                if value is not None:
                    # Add the value multiple times based on its weight
                    weighted_values.extend([value] * int(weight * 1000))  # Scale up for better precision
            
            if weighted_values:
                if param == 'hidden_layer_size':
                    # Hidden layer size should be integer
                    param_analysis[param] = {
                        'recommended': int(np.median(weighted_values)),
                        'weighted_mean': np.mean(weighted_values),
                        'std': np.std(weighted_values),
                        'min': np.min(weighted_values),
                        'max': np.max(weighted_values),
                        'count': len(weighted_values)
                    }
                else:
                    param_analysis[param] = {
                        'recommended': np.median(weighted_values),
                        'weighted_mean': np.mean(weighted_values),
                        'std': np.std(weighted_values),
                        'min': np.min(weighted_values),
                        'max': np.max(weighted_values),
                        'count': len(weighted_values)
                    }
        
        return param_analysis
    
    def _generate_optimal_configs(self, results):
        """Generate optimal_configs.json with complete parameter sets for each encoding scheme."""
        optimal_configs = {}
        
        for encoding_scheme, data in results.items():
            if 'recommended_config' not in data or not data['recommended_config']:
                continue
                
            config = data['recommended_config']
            
            # Build the optimal configuration
            optimal_config = {
                "model_params": {
                    "num_layers": config.get('num_layers', {}).get('recommended', 1),
                    "hidden_layer_size": config.get('hidden_layer_size', {}).get('recommended', 1024),
                    "dropout_rate": config.get('dropout_rate', {}).get('recommended', 0.2),
                    "activation_fn": config.get('activation_fn', {}).get('recommended', 'selu')
                },
                "optimizer": {
                    "name": config.get('optimizer_name', {}).get('recommended', 'RMSprop'),
                    "lr": config.get('optimizer_lr', {}).get('recommended', 0.001)
                },
                "loss_fn": config.get('loss_fn', {}).get('recommended', 'BCEWithLogitsLoss'),
                "threshold": config.get('threshold', {}).get('recommended', 0.3),
                "batch_size": config.get('batch_size', {}).get('recommended', 8),
                "lr_scheduler": {
                    "name": config.get('lr_scheduler_name', {}).get('recommended', 'None')
                }
            }
            
            # Add optimizer-specific parameters
            optimizer_name = optimal_config["optimizer"]["name"]
            if optimizer_name == "SGD":
                optimal_config["optimizer"]["momentum"] = config.get('optimizer_momentum', {}).get('recommended', 0.9)
            elif optimizer_name in ["Adam", "AdamW"]:
                optimal_config["optimizer"]["weight_decay"] = config.get('optimizer_weight_decay', {}).get('recommended', 0.0)
                optimal_config["optimizer"]["eps"] = config.get('optimizer_eps', {}).get('recommended', 1e-8)
                if optimizer_name == "Adam":
                    optimal_config["optimizer"]["betas"] = (0.9, 0.999)  # Default values
            elif optimizer_name == "RMSprop":
                optimal_config["optimizer"]["alpha"] = config.get('optimizer_alpha', {}).get('recommended', 0.99)
            
            # Add scheduler-specific parameters
            scheduler_name = optimal_config["lr_scheduler"]["name"]
            if scheduler_name == "ReduceLROnPlateau":
                optimal_config["lr_scheduler"]["mode"] = config.get('lr_scheduler_mode', {}).get('recommended', 'min')
                optimal_config["lr_scheduler"]["factor"] = config.get('lr_scheduler_factor', {}).get('recommended', 0.1)
                optimal_config["lr_scheduler"]["patience"] = config.get('lr_scheduler_patience', {}).get('recommended', 10)
            elif scheduler_name == "CosineAnnealingLR":
                optimal_config["lr_scheduler"]["T_max"] = config.get('lr_scheduler_T_max', {}).get('recommended', 10)
                optimal_config["lr_scheduler"]["eta_min"] = config.get('lr_scheduler_eta_min', {}).get('recommended', 0)
            elif scheduler_name == "CyclicLR":
                optimal_config["lr_scheduler"]["base_lr"] = config.get('lr_scheduler_base_lr', {}).get('recommended', 1e-5)
                optimal_config["lr_scheduler"]["max_lr"] = config.get('lr_scheduler_max_lr', {}).get('recommended', 1e-1)
                optimal_config["lr_scheduler"]["step_size_up"] = config.get('lr_scheduler_step_size_up', {}).get('recommended', 2000)
                optimal_config["lr_scheduler"]["mode"] = config.get('lr_scheduler_mode_cyclic', {}).get('recommended', 'triangular')
            elif scheduler_name == "StepLR":
                optimal_config["lr_scheduler"]["step_size"] = config.get('lr_scheduler_step_size', {}).get('recommended', 30)
                optimal_config["lr_scheduler"]["gamma"] = config.get('lr_scheduler_gamma', {}).get('recommended', 0.1)
            
            optimal_configs[encoding_scheme] = optimal_config
            
        return optimal_configs
    
    def _generate_narrow_search_spaces(self, results):
        """Generate narrow Ray Tune search spaces based on top 15% performers."""
        search_spaces = {}
        
        for encoding_scheme, data in results.items():
            if 'recommended_config' not in data or not data['recommended_config']:
                continue
                
            config = data['recommended_config']
            
            # Get top 15% performers for range calculations
            experiments = self.data[encoding_scheme]
            hyperparams = self._extract_hyperparameters(experiments)
            sorted_hp = sorted(hyperparams, key=lambda x: x['performance']['average_dice'], reverse=True)
            top_performers = sorted_hp[:max(1, int(len(sorted_hp) * 0.15))]
            
            # Extract ranges from top performers
            ranges = self._calculate_parameter_ranges(top_performers)
            
            # Build narrow search space
            search_space = {
                "output_dim": "len(all_two_grams)",  # This will be set dynamically
                "num_layers": self._create_tune_choice("num_layers", ranges),
                "hidden_layer_size": self._create_tune_choice("hidden_layer_size", ranges),
                "dropout_rate": self._create_tune_uniform("dropout_rate", ranges),
                "activation_fn": self._create_tune_choice("activation_fn", ranges),
                "optimizer": self._create_optimizer_choice(ranges),
                "loss_fn": self._create_tune_choice("loss_fn", ranges),
                "threshold": self._create_tune_uniform("threshold", ranges),
                "lr_scheduler": self._create_scheduler_choice(ranges),
                "batch_size": self._create_tune_choice("batch_size", ranges),
            }
            
            search_spaces[encoding_scheme] = {
                "search_space": search_space,
                "parameter_ranges": ranges,
                "top_performers_count": len(top_performers),
                "total_experiments": len(experiments)
            }
            
        return search_spaces
    
    def _calculate_parameter_ranges(self, top_performers):
        """Calculate parameter ranges from top performers."""
        ranges = {}
        
        # Collect all values for each parameter
        param_values = defaultdict(list)
        for hp in top_performers:
            for param, value in hp.items():
                if value is not None and param not in ['performance', 'dataset']:
                    # Skip nested dictionaries (like optimizer, lr_scheduler)
                    if not isinstance(value, dict):
                        param_values[param].append(value)
        
        # Calculate ranges
        for param, values in param_values.items():
            if not values:
                continue
                
            if param in ['num_layers', 'hidden_layer_size', 'batch_size', 'lr_scheduler_patience', 
                        'lr_scheduler_step_size_up', 'lr_scheduler_step_size']:
                # Integer parameters
                ranges[param] = {
                    'type': 'integer',
                    'values': sorted(list(set(values))),
                    'min': min(values),
                    'max': max(values)
                }
            elif param in ['activation_fn', 'optimizer_name', 'loss_fn', 'lr_scheduler_name', 
                          'lr_scheduler_mode', 'lr_scheduler_mode_cyclic']:
                # Categorical parameters
                ranges[param] = {
                    'type': 'categorical',
                    'values': sorted(list(set(values))),
                    'count': len(set(values))
                }
            else:
                # Continuous parameters
                ranges[param] = {
                    'type': 'continuous',
                    'values': values,
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
        
        return ranges
    
    def _create_tune_choice(self, param, ranges):
        """Create tune.choice for categorical parameters."""
        if param not in ranges or ranges[param]['type'] not in ['categorical', 'integer']:
            return f"# Missing data for {param}"
        
        values = ranges[param]['values']
        if len(values) == 1:
            return f"tune.choice([{values[0]}])  # Only one value found"
        else:
            return f"tune.choice({values})"
    
    def _create_tune_uniform(self, param, ranges):
        """Create tune.uniform for continuous parameters."""
        if param not in ranges or ranges[param]['type'] != 'continuous':
            return f"# Missing data for {param}"
        
        min_val = ranges[param]['min']
        max_val = ranges[param]['max']
        
        # Add some padding to the range (10% on each side)
        padding = (max_val - min_val) * 0.1
        min_val = max(0, min_val - padding)  # Ensure non-negative for most parameters
        max_val = max_val + padding
        
        return f"tune.uniform({min_val:.6f}, {max_val:.6f})"
    
    def _create_optimizer_choice(self, ranges):
        """Create optimizer choice with learning rate ranges."""
        if 'optimizer_name' not in ranges:
            return "# Missing optimizer data"
        
        optimizer_names = ranges['optimizer_name']['values']
        optimizer_choices = []
        
        for opt_name in optimizer_names:
            if 'optimizer_lr' in ranges:
                lr_min = ranges['optimizer_lr']['min']
                lr_max = ranges['optimizer_lr']['max']
                # Add padding for learning rate
                lr_padding = (lr_max - lr_min) * 0.2
                lr_min = max(1e-6, lr_min - lr_padding)
                lr_max = lr_max + lr_padding
                
                optimizer_config = f'{{"name": "{opt_name}", "lr": tune.loguniform({lr_min:.2e}, {lr_max:.2e})}}'
            else:
                optimizer_config = f'{{"name": "{opt_name}", "lr": tune.loguniform(1e-5, 1e-3)}}'
            
            optimizer_choices.append(optimizer_config)
        
        return "tune.choice([\n    " + ",\n    ".join(optimizer_choices) + "\n])"
    
    def _create_scheduler_choice(self, ranges):
        """Create scheduler choice with parameter ranges."""
        if 'lr_scheduler_name' not in ranges:
            return "# Missing scheduler data"
        
        scheduler_names = ranges['lr_scheduler_name']['values']
        scheduler_choices = []
        
        for sched_name in scheduler_names:
            if sched_name == "None":
                scheduler_choices.append('{"name": "None"}')
            elif sched_name == "ReduceLROnPlateau":
                # Get ranges for ReduceLROnPlateau parameters
                factor_min = ranges.get('lr_scheduler_factor', {}).get('min', 0.1)
                factor_max = ranges.get('lr_scheduler_factor', {}).get('max', 0.5)
                patience_vals = ranges.get('lr_scheduler_patience', {}).get('values', [5, 10, 15])
                
                scheduler_config = f'{{"name": "ReduceLROnPlateau", "mode": "min", "factor": tune.uniform({factor_min:.3f}, {factor_max:.3f}), "patience": tune.choice({patience_vals})}}'
                scheduler_choices.append(scheduler_config)
                
            elif sched_name == "CosineAnnealingLR":
                t_max_min = ranges.get('lr_scheduler_T_max', {}).get('min', 10)
                t_max_max = ranges.get('lr_scheduler_T_max', {}).get('max', 50)
                eta_min_vals = ranges.get('lr_scheduler_eta_min', {}).get('values', [1e-5, 1e-6, 0])
                
                scheduler_config = f'{{"name": "CosineAnnealingLR", "T_max": tune.loguniform({t_max_min:.1f}, {t_max_max:.1f}), "eta_min": tune.choice({eta_min_vals})}}'
                scheduler_choices.append(scheduler_config)
                
            elif sched_name == "CyclicLR":
                base_lr_min = ranges.get('lr_scheduler_base_lr', {}).get('min', 1e-5)
                base_lr_max = ranges.get('lr_scheduler_base_lr', {}).get('max', 1e-3)
                max_lr_min = ranges.get('lr_scheduler_max_lr', {}).get('min', 1e-3)
                max_lr_max = ranges.get('lr_scheduler_max_lr', {}).get('max', 1e-1)
                step_size_vals = ranges.get('lr_scheduler_step_size_up', {}).get('values', [2000, 4000])
                mode_vals = ranges.get('lr_scheduler_mode_cyclic', {}).get('values', ["triangular", "triangular2", "exp_range"])
                
                scheduler_config = f'{{"name": "CyclicLR", "base_lr": tune.loguniform({base_lr_min:.2e}, {base_lr_max:.2e}), "max_lr": tune.loguniform({max_lr_min:.2e}, {max_lr_max:.2e}), "step_size_up": tune.choice({step_size_vals}), "mode_cyclic": tune.choice({mode_vals})}}'
                scheduler_choices.append(scheduler_config)
                
            elif sched_name == "StepLR":
                step_size_vals = ranges.get('lr_scheduler_step_size', {}).get('values', [30])
                gamma_min = ranges.get('lr_scheduler_gamma', {}).get('min', 0.1)
                gamma_max = ranges.get('lr_scheduler_gamma', {}).get('max', 0.9)
                
                scheduler_config = f'{{"name": "StepLR", "step_size": tune.choice({step_size_vals}), "gamma": tune.uniform({gamma_min:.3f}, {gamma_max:.3f})}}'
                scheduler_choices.append(scheduler_config)
        
        return "tune.choice([\n    " + ",\n    ".join(scheduler_choices) + "\n])"
    
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
            'optimizer_momentum': {'type': 'continuous', 'buckets': [0.0, 0.5, 0.9, 0.95, 0.99]},
            'optimizer_weight_decay': {'type': 'continuous', 'buckets': [0.0, 1e-4, 1e-3, 1e-2]},
            'optimizer_eps': {'type': 'continuous', 'buckets': [1e-8, 1e-7, 1e-6, 1e-5]},
            'optimizer_alpha': {'type': 'continuous', 'buckets': [0.9, 0.95, 0.99, 0.999]},
            'loss_fn': {'type': 'categorical'},
            'threshold': {'type': 'continuous', 'buckets': [0.1, 0.2, 0.3, 0.4, 0.5]},
            'batch_size': {'type': 'categorical'},
            'lr_scheduler_name': {'type': 'categorical'},
            'lr_scheduler_mode': {'type': 'categorical'},
            'lr_scheduler_factor': {'type': 'continuous', 'buckets': [0.1, 0.2, 0.3, 0.4, 0.5]},
            'lr_scheduler_patience': {'type': 'categorical'},
            'lr_scheduler_T_max': {'type': 'continuous', 'buckets': [10, 20, 50, 100]},
            'lr_scheduler_eta_min': {'type': 'continuous', 'buckets': [0, 1e-6, 1e-5, 1e-4]},
            'lr_scheduler_base_lr': {'type': 'continuous', 'buckets': [1e-5, 1e-4, 1e-3]},
            'lr_scheduler_max_lr': {'type': 'continuous', 'buckets': [1e-3, 1e-2, 1e-1]},
            'lr_scheduler_step_size_up': {'type': 'categorical'},
            'lr_scheduler_mode_cyclic': {'type': 'categorical'},
            'lr_scheduler_gamma': {'type': 'continuous', 'buckets': [0.1, 0.2, 0.5, 0.9]},
            'lr_scheduler_step_size': {'type': 'categorical'}
        }
        
        for param, config in param_configs.items():
            values = []
            performances = []
            
            for hp in hyperparams:
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
        """Generate a comprehensive report with detailed parameter analysis."""
        report = []
        report.append("# Hyperparameter Analysis Report - Detailed Parameter Analysis")
        report.append("=" * 80)
        report.append("Based on top 15% performers with performance-weighted analysis")
        report.append("=" * 80)
        
        for encoding_scheme, data in results.items():
            report.append(f"\n## {encoding_scheme} - Recommended Configuration")
            report.append(f"Total Experiments Analyzed: {data['total_experiments']}")
            report.append("-" * 60)
            
            # Recommended configuration with weighted analysis
            if 'recommended_config' in data and data['recommended_config']:
                report.append("\n### Model Architecture Parameters:")
                report.append("")
                model_params = ['num_layers', 'hidden_layer_size', 'dropout_rate', 'activation_fn']
                for param in model_params:
                    if param in data['recommended_config']:
                        config = data['recommended_config'][param]
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
                
                report.append("\n### Optimizer Configuration:")
                report.append("")
                optimizer_params = ['optimizer_name', 'optimizer_lr', 'optimizer_momentum', 'optimizer_weight_decay', 'optimizer_eps', 'optimizer_alpha']
                for param in optimizer_params:
                    if param in data['recommended_config']:
                        config = data['recommended_config'][param]
                        if 'recommended' in config:
                            if isinstance(config['recommended'], (int, float)):
                                if 'weighted_score' in config:
                                    report.append(f"  • **{param}**: {config['recommended']:.6f} (weighted score: {config['weighted_score']:.3f})")
                                else:
                                    report.append(f"  • **{param}**: {config['recommended']:.6f} (weighted median, range: {config['min']:.6f}-{config['max']:.6f})")
                            else:
                                if 'weighted_score' in config:
                                    report.append(f"  • **{param}**: {config['recommended']} (weighted score: {config['weighted_score']:.3f})")
                                else:
                                    report.append(f"  • **{param}**: {config['recommended']}")
                
                report.append("\n### Learning Rate Scheduler Configuration:")
                report.append("")
                scheduler_params = ['lr_scheduler_name', 'lr_scheduler_mode', 'lr_scheduler_factor', 'lr_scheduler_patience', 
                                  'lr_scheduler_T_max', 'lr_scheduler_eta_min', 'lr_scheduler_base_lr', 'lr_scheduler_max_lr',
                                  'lr_scheduler_step_size_up', 'lr_scheduler_mode_cyclic', 'lr_scheduler_gamma', 'lr_scheduler_step_size']
                for param in scheduler_params:
                    if param in data['recommended_config']:
                        config = data['recommended_config'][param]
                        if 'recommended' in config:
                            if isinstance(config['recommended'], (int, float)):
                                if 'weighted_score' in config:
                                    report.append(f"  • **{param}**: {config['recommended']:.6f} (weighted score: {config['weighted_score']:.3f})")
                                else:
                                    report.append(f"  • **{param}**: {config['recommended']:.6f} (weighted median, range: {config['min']:.6f}-{config['max']:.6f})")
                            else:
                                if 'weighted_score' in config:
                                    report.append(f"  • **{param}**: {config['recommended']} (weighted score: {config['weighted_score']:.3f})")
                                else:
                                    report.append(f"  • **{param}**: {config['recommended']}")
                
                report.append("\n### Training Configuration:")
                report.append("")
                training_params = ['loss_fn', 'threshold', 'batch_size']
                for param in training_params:
                    if param in data['recommended_config']:
                        config = data['recommended_config'][param]
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
                report.append("- Top 15% of experiments by average_dice score are analyzed")
                report.append("- Each configuration is weighted by its performance relative to others")
                report.append("- Higher performing configurations have proportionally greater influence")
                report.append("- Categorical parameters: most frequent weighted choice")
                report.append("- Continuous parameters: weighted median/mean")
                report.append("- Optimizer and scheduler parameters are analyzed individually for complete configuration")
            else:
                report.append("\nNo recommended configuration available (insufficient data)")
            
            # Add narrow search space section
            if 'search_spaces' in data and data['search_spaces']:
                report.append("\n" + "="*80)
                report.append(f"## {encoding_scheme} - Narrow Ray Tune Search Space")
                report.append("="*80)
                report.append(f"Based on top {data['search_spaces']['top_performers_count']} performers (top 15%)")
                report.append(f"Total experiments analyzed: {data['search_spaces']['total_experiments']}")
                report.append("")
                
                search_space = data['search_spaces']['search_space']
                report.append("### Ray Tune Search Space Code:")
                report.append("```python")
                report.append("search_space = {")
                for param, value in search_space.items():
                    if param == "output_dim":
                        report.append(f'    "{param}": {value},')
                    else:
                        # Format the value for better readability
                        if isinstance(value, str) and value.startswith("tune."):
                            # Multi-line formatting for complex tune choices
                            if "tune.choice([\n" in value:
                                lines = value.split('\n')
                                report.append(f'    "{param}": {lines[0]}')
                                for line in lines[1:-1]:
                                    report.append(f'        {line}')
                                report.append(f'    {lines[-1]},')
                            else:
                                report.append(f'    "{param}": {value},')
                        else:
                            report.append(f'    "{param}": {value},')
                report.append("}")
                report.append("```")
                
                report.append("\n### Parameter Range Analysis:")
                ranges = data['search_spaces']['parameter_ranges']
                for param, range_info in ranges.items():
                    if range_info['type'] == 'categorical':
                        report.append(f"  • **{param}**: {range_info['values']} ({range_info['count']} unique values)")
                    elif range_info['type'] == 'integer':
                        report.append(f"  • **{param}**: {range_info['values']} (range: {range_info['min']}-{range_info['max']})")
                    else:  # continuous
                        # Handle potential tuple values
                        min_val = range_info['min']
                        max_val = range_info['max']
                        mean_val = range_info['mean']
                        std_val = range_info['std']
                        
                        if isinstance(min_val, (int, float)):
                            report.append(f"  • **{param}**: range {min_val:.6f}-{max_val:.6f} (mean: {mean_val:.6f}, std: {std_val:.6f})")
                        else:
                            report.append(f"  • **{param}**: range {min_val}-{max_val} (mean: {mean_val}, std: {std_val})")
        
        return "\n".join(report)
    
    def save_results(self, results, output_dir="analysis"):
        """Save the analysis report, optimal configs JSON, and search spaces JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed analysis report
        report = self.generate_report(results)
        with open(output_dir / "hyperparameter_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Generate and save optimal configs JSON
        optimal_configs = self._generate_optimal_configs(results)
        with open(output_dir / "optimal_configs.json", 'w', encoding='utf-8') as f:
            json.dump(optimal_configs, f, indent=4)
        
        # Extract and save search spaces JSON
        search_spaces = {}
        for encoding_scheme, data in results.items():
            if 'search_spaces' in data:
                search_spaces[encoding_scheme] = data['search_spaces']
        
        if search_spaces:
            with open(output_dir / "narrow_search_spaces.json", 'w', encoding='utf-8') as f:
                json.dump(search_spaces, f, indent=4)
        
        print(f"\nAnalysis report saved to {output_dir}/hyperparameter_analysis_report.txt")
        print(f"Optimal configs saved to {output_dir}/optimal_configs.json")
        if search_spaces:
            print(f"Narrow search spaces saved to {output_dir}/narrow_search_spaces.json")
    

def main():
    """Main function to run the hyperparameter analysis with narrow search space generation."""
    print("Starting Weighted Hyperparameter Analysis with Narrow Search Space Generation...")
    
    # Initialize analyzer
    analyzer = HyperparameterAnalyzer()
    
    # Load data
    analyzer.load_experiment_data()
    
    # Analyze hyperparameters and generate search spaces
    results = analyzer.analyze_hyperparameters()
    
    # Save results (report, optimal configs, and search spaces)
    analyzer.save_results(results)
    
    print("Analysis complete! Check the analysis directory for:")
    print("  - hyperparameter_analysis_report.txt (detailed analysis with search spaces)")
    print("  - optimal_configs.json (recommended configurations)")
    print("  - narrow_search_spaces.json (Ray Tune search spaces for each encoding)")

if __name__ == "__main__":
    main()