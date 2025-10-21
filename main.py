#!/usr/bin/env python3
"""
Main entry point for the Attack.
"""

import json
import os
import sys
import argparse
from typing import Dict, Any
from nepal import run_nepal


def load_config(config_path: str = "nepal_config.json") -> Dict[str, Any]:
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
    


def main():
    """
    Command line interface for running NEPAL experiments.
    
    Usage:
        python main.py [--config CONFIG_PATH]
    """
    parser = argparse.ArgumentParser(
        description="NEPAL - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run with default config file (nepal_config.json)
    python main.py
    
    # Run with custom config file
    python main.py --config my_config.json
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="nepal_config.json",
        help="Path to configuration file (default: nepal_config.json)"
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
        NEPAL_CONFIG = config["NEPAL_CONFIG"]
        
        # Override verbose setting if specified
        if args.verbose:
            GLOBAL_CONFIG["Verbose"] = True
        
        if args.verbose:
            print(f"[INFO] GMA enabled: {GLOBAL_CONFIG['GraphMatchingAttack']}")
            print(f"[INFO] Parallel Trials: {NEPAL_CONFIG.get('ParallelTrials', 0)}")
            print(f"[INFO] GPU Usage: {GLOBAL_CONFIG.get('UseGPU', False)}")
            print(f"[INFO] GPU Count: {GLOBAL_CONFIG.get('GPUCount', 0)}")
            print(f"[INFO] Dataset: {GLOBAL_CONFIG.get('Data', 'Not specified')}")
            print(f"[INFO] Encoding Algorithm: {ENC_CONFIG.get('AliceAlgo', 'Not specified')}")
        
        # Run the experiment
        exit_code = run_nepal(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, NEPAL_CONFIG)
        
        
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
