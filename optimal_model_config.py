#!/usr/bin/env python3
"""
Optimal Model Configuration based on Hyperparameter Analysis

This module provides functions to initialize the optimal ANN configuration
for each encoding scheme based on the hyperparameter analysis results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from pytorch_models.bloomfilter_model import BloomFilterModel
from pytorch_models.tabminhash_model import TabMinHashModel
from pytorch_models.twostephash_model import TwoStepHashModel

# Optimal configurations based on hyperparameter analysis
OPTIMAL_CONFIGS = {
    "BloomFilter": {
        "model_params": {
            "num_layers": 1,
            "hidden_layer_size": 4096,
            "dropout_rate": 0.2253,
            "activation_fn": "selu"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 0.0002
        },
        "loss_fn": "BCEWithLogitsLoss",
        "threshold": 0.3294,
        "batch_size": 8,
        "lr_scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": 100,
            "eta_min": 1e-6
        }
    },
    "TabMinHash": {
        "model_params": {
            "num_layers": 1,
            "hidden_layer_size": 4096,
            "dropout_rate": 0.2888,
            "activation_fn": "selu"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 0.0001
        },
        "loss_fn": "BCEWithLogitsLoss",
        "threshold": 0.3355,
        "batch_size": 8,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5
        }
    },
    "TwoStepHash": {
        "model_params": {
            "num_layers": 1,
            "hidden_layer_size": 3072,  # Rounded from 3072.0000
            "dropout_rate": 0.2218,
            "activation_fn": "elu"
        },
        "optimizer": {
            "name": "RMSprop",
            "lr": 0.0002
        },
        "loss_fn": "MultiLabelSoftMarginLoss",
        "threshold": 0.2322,
        "batch_size": 8,
        "lr_scheduler": {
            "name": "None"  # No scheduler for TwoStepHash
        }
    }
}

def get_optimal_model(encoding_scheme, input_dim, output_dim, device=None):
    """
    Initialize the optimal model for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme ("BloomFilter", "TabMinHash", or "TwoStepHash")
        input_dim (int): Input dimension of the model
        output_dim (int): Output dimension of the model
        device (torch.device, optional): Device to move the model to
        
    Returns:
        torch.nn.Module: The initialized model
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    config = OPTIMAL_CONFIGS[encoding_scheme]
    model_params = config["model_params"]
    
    # Initialize the appropriate model
    if encoding_scheme == "BloomFilter":
        model = BloomFilterModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer=model_params["hidden_layer_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"],
            activation_fn=model_params["activation_fn"]
        )
    elif encoding_scheme == "TabMinHash":
        model = TabMinHashModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer=model_params["hidden_layer_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"],
            activation_fn=model_params["activation_fn"]
        )
    elif encoding_scheme == "TwoStepHash":
        model = TwoStepHashModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layer=model_params["hidden_layer_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"],
            activation_fn=model_params["activation_fn"]
        )
    
    if device is not None:
        model = model.to(device)
    
    return model

def get_optimal_optimizer(model, encoding_scheme):
    """
    Initialize the optimal optimizer for the given model and encoding scheme.
    
    Args:
        model (torch.nn.Module): The model to optimize
        encoding_scheme (str): The encoding scheme
        
    Returns:
        torch.optim.Optimizer: The initialized optimizer
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    config = OPTIMAL_CONFIGS[encoding_scheme]
    optimizer_config = config["optimizer"]
    
    if optimizer_config["name"] == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=optimizer_config["lr"])
    elif optimizer_config["name"] == "Adam":
        return optim.Adam(model.parameters(), lr=optimizer_config["lr"])
    elif optimizer_config["name"] == "AdamW":
        return optim.AdamW(model.parameters(), lr=optimizer_config["lr"])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")

def get_optimal_loss_function(encoding_scheme):
    """
    Get the optimal loss function for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
        
    Returns:
        torch.nn.Module: The loss function
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    config = OPTIMAL_CONFIGS[encoding_scheme]
    loss_fn_name = config["loss_fn"]
    
    if loss_fn_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_fn_name == "MultiLabelSoftMarginLoss":
        return nn.MultiLabelSoftMarginLoss()
    elif loss_fn_name == "SoftMarginLoss":
        return nn.SoftMarginLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")

def get_optimal_scheduler(optimizer, encoding_scheme):
    """
    Initialize the optimal learning rate scheduler for the given optimizer and encoding scheme.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer
        encoding_scheme (str): The encoding scheme
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The scheduler or None
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    config = OPTIMAL_CONFIGS[encoding_scheme]
    scheduler_config = config["lr_scheduler"]
    
    if scheduler_config["name"] == "None":
        return None
    elif scheduler_config["name"] == "CyclicLR":
        return CyclicLR(
            optimizer,
            base_lr=scheduler_config.get("base_lr", 1e-6),
            max_lr=scheduler_config.get("max_lr", 1e-3),
            step_size_up=scheduler_config.get("step_size_up", 2000),
            mode=scheduler_config.get("mode", "triangular")
        )
    elif scheduler_config["name"] == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 1e-6)
        )
    elif scheduler_config["name"] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")

def get_optimal_threshold(encoding_scheme):
    """
    Get the optimal threshold for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
        
    Returns:
        float: The optimal threshold
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    return OPTIMAL_CONFIGS[encoding_scheme]["threshold"]

def get_optimal_batch_size(encoding_scheme):
    """
    Get the optimal batch size for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
        
    Returns:
        int: The optimal batch size
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    return OPTIMAL_CONFIGS[encoding_scheme]["batch_size"]

def get_optimal_config(encoding_scheme):
    """
    Get the complete optimal configuration for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
        
    Returns:
        dict: The complete configuration
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")
    
    return OPTIMAL_CONFIGS[encoding_scheme].copy()

def initialize_optimal_training_setup(encoding_scheme, input_dim, output_dim, device=None):
    """
    Initialize the complete optimal training setup for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
        input_dim (int): Input dimension of the model
        output_dim (int): Output dimension of the model
        device (torch.device, optional): Device to use
        
    Returns:
        dict: Dictionary containing model, optimizer, criterion, scheduler, threshold, and batch_size
    """
    # Initialize model
    model = get_optimal_model(encoding_scheme, input_dim, output_dim, device)
    
    # Initialize optimizer
    optimizer = get_optimal_optimizer(model, encoding_scheme)
    
    # Initialize loss function
    criterion = get_optimal_loss_function(encoding_scheme)
    
    # Initialize scheduler
    scheduler = get_optimal_scheduler(optimizer, encoding_scheme)
    
    # Get optimal threshold and batch size
    threshold = get_optimal_threshold(encoding_scheme)
    batch_size = get_optimal_batch_size(encoding_scheme)
    
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "threshold": threshold,
        "batch_size": batch_size
    }

def print_optimal_config(encoding_scheme):
    """
    Print the optimal configuration for the given encoding scheme.
    
    Args:
        encoding_scheme (str): The encoding scheme
    """
    if encoding_scheme not in OPTIMAL_CONFIGS:
        print(f"Unknown encoding scheme: {encoding_scheme}")
        return
    
    config = OPTIMAL_CONFIGS[encoding_scheme]
    print(f"\n=== Optimal Configuration for {encoding_scheme} ===")
    print(f"Model Parameters:")
    for key, value in config["model_params"].items():
        print(f"  {key}: {value}")
    print(f"Optimizer: {config['optimizer']['name']} (lr={config['optimizer']['lr']})")
    print(f"Loss Function: {config['loss_fn']}")
    print(f"Threshold: {config['threshold']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"LR Scheduler: {config['lr_scheduler']['name']}")
    if config['lr_scheduler']['name'] != "None":
        for key, value in config['lr_scheduler'].items():
            if key != "name":
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # Print optimal configurations for all encoding schemes
    for scheme in ["BloomFilter", "TabMinHash", "TwoStepHash"]:
        print_optimal_config(scheme)
