#!/usr/bin/env python3
"""
Main entry point for the Dataset Extension Attack (DEA) system.

This script provides a unified interface to run the combined dea.py module
with or without hyperparameter optimization.

The choice is determined by the DEA_CONFIG["HPO"] setting in dea_config.json.
"""

import json
import os
import sys
import argparse
from typing import Dict, Any


def load_config(config_path: str = "dea_config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing all configuration sections
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def run_dea(GLOBAL_CONFIG: Dict[str, Any], 
           ENC_CONFIG: Dict[str, Any], 
           EMB_CONFIG: Dict[str, Any], 
           ALIGN_CONFIG: Dict[str, Any], 
           DEA_CONFIG: Dict[str, Any]) -> int:
    """
    Main entry point for running DEA experiments.
    
    Args:
        GLOBAL_CONFIG: Global configuration parameters
        ENC_CONFIG: Encoding configuration parameters
        EMB_CONFIG: Embedding configuration parameters
        ALIGN_CONFIG: Alignment configuration parameters
        DEA_CONFIG: DEA-specific configuration parameters
        
    Returns:
        Exit code (0 for success)
    """
    # Import the combined DEA module
    from dea import run_dea as run_dea_combined
    
    # Determine whether to skip HPO based on config setting
    skip_hpo = not DEA_CONFIG.get("HPO", True)
    
    if skip_hpo:
        print("[INFO] Running without Hyperparameter Optimization (Skip HPO)")
    else:
        print("[INFO] Running with Hyperparameter Optimization (HPO)")
    
    # Run the combined DEA function with the skip_hpo parameter
    return run_dea_combined(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG, skip_hpo=skip_hpo)


def main():
    """
    Command line interface for running DEA experiments.
    
    Usage:
        python main.py [--config CONFIG_PATH]
    """
    parser = argparse.ArgumentParser(
        description="Dataset Extension Attack (DEA) - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config file (dea_config.json)
    python main.py
    
    # Run with custom config file
    python main.py --config my_config.json
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="dea_config.json",
        help="Path to configuration file (default: dea_config.json)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.verbose:
            print(f"[INFO] Loading configuration from: {args.config}")
        
        config = load_config(args.config)
        
        # Extract configuration sections
        GLOBAL_CONFIG = config["GLOBAL_CONFIG"]
        ENC_CONFIG = config["ENC_CONFIG"]
        EMB_CONFIG = config["EMB_CONFIG"]
        ALIGN_CONFIG = config["ALIGN_CONFIG"]
        DEA_CONFIG = config["DEA_CONFIG"]
        
        # Override verbose setting if specified
        if args.verbose:
            GLOBAL_CONFIG["Verbose"] = True
        
        if args.verbose:
            hpo_enabled = DEA_CONFIG.get('HPO', True)
            print(f"[INFO] HPO enabled: {hpo_enabled}")
            print(f"[INFO] Skip HPO: {not hpo_enabled}")
            print(f"[INFO] Parallel Trials: {DEA_CONFIG.get('ParallelTrials', 4)}")
            print(f"[INFO] GPU Usage: {GLOBAL_CONFIG.get('UseGPU', False)}")
            print(f"[INFO] GPU Count: {GLOBAL_CONFIG.get('GPUCount', 0)}")
            print(f"[INFO] Dataset: {GLOBAL_CONFIG.get('Data', 'Not specified')}")
            print(f"[INFO] Encoding Algorithm: {ENC_CONFIG.get('AliceAlgo', 'Not specified')}")
        
        # Run the experiment
        exit_code = run_dea(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, DEA_CONFIG)
        
        if args.verbose:
            print(f"[INFO] Experiment completed with exit code: {exit_code}")
        
        return exit_code
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except KeyError as e:
        print(f"[ERROR] Missing required configuration section: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
