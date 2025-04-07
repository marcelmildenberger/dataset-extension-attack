import torch.nn as nn

activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
}

class BaseModel(nn.Module):
    def __init__(self, input_dim, num_two_grams, num_layers, hidden_layer_size, dropout_rate, activation_fn):
        super(BaseModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_size)) # Input layer
        layers.append(activation_functions[activation_fn])
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):  # Add dynamic hidden layers
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(activation_functions[activation_fn])
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layer_size, num_two_grams))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)