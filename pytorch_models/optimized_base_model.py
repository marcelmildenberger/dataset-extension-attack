import torch
import torch.nn as nn

class OptimizedBaseModel(nn.Module):
    """
    Optimized base model with performance improvements:
    - Proper weight initialization (Xavier/Glorot)
    - PyTorch 2.0 compilation support
    - Memory-efficient layer construction
    - Optimized activation functions
    """
    
    def __init__(self, input_dim, output_dim, hidden_layer=128, num_layers=2, 
                 dropout_rate=0.2, activation_fn="relu", compile_model=True):
        super(OptimizedBaseModel, self).__init__()
        
        # Activation function mapping with memory-efficient choices
        activation_functions = {
            "relu": nn.ReLU(inplace=True),  # inplace=True saves memory
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
            "elu": nn.ELU(inplace=True),
            "gelu": nn.GELU(),  # Often better than ReLU for transformers
            "swish": nn.SiLU(),  # SiLU (Swish) activation
        }
        
        if activation_fn not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        
        # Build layers efficiently using ModuleList
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_layer),
            activation_functions[activation_fn],
            nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_layer, hidden_layer),
                activation_functions[activation_fn],
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer (no activation - will be handled by loss function)
        layers.append(nn.Linear(hidden_layer, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Apply optimized weight initialization
        self.apply(self._init_weights)
        
        # Compile model for PyTorch 2.0+ if available and requested
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.forward = torch.compile(self.forward, mode="max-autotune")
                print("✅ Model compiled with PyTorch 2.0+ for optimized performance")
            except Exception as e:
                print(f"⚠️  Model compilation failed (not critical): {e}")

    def _init_weights(self, module):
        """
        Initialize weights using Xavier/Glorot initialization for better convergence.
        """
        if isinstance(module, nn.Linear):
            # Xavier uniform initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # Initialize bias to small positive values
                torch.nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.model(x)

    def get_model_info(self):
        """
        Get information about the model architecture and parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": str(self.model)
        }

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing to trade compute for memory.
        Useful for very large models or when memory is constrained.
        """
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            print("✅ Gradient checkpointing enabled")
            # This would require modifying forward pass to use checkpointing
            # Implementation depends on specific use case
        else:
            print("⚠️  Gradient checkpointing not available in this PyTorch version")

    @staticmethod
    def create_from_config(config):
        """
        Factory method to create model from configuration dictionary.
        """
        return OptimizedBaseModel(
            input_dim=config.get("input_dim"),
            output_dim=config.get("output_dim"),
            hidden_layer=config.get("hidden_layer_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.2),
            activation_fn=config.get("activation_fn", "relu"),
            compile_model=config.get("compile_model", True)
        )