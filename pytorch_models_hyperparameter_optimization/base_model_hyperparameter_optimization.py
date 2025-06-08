import torch.nn as nn

activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
}

class BaseModelHyperparameterOptimization(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_layer_size, dropout_rate, activation_fn):
        super(BaseModelHyperparameterOptimization, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_size)) # Input layer
        layers.append(activation_functions[activation_fn])
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):  # Add dynamic hidden layers
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(activation_functions[activation_fn])
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layer_size, output_dim))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)