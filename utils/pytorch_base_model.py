import torch.nn as nn

activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
}

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer=256, num_layers=4, dropout_rate=0.2, activation_fn="relu"):
        super(BaseModel, self).__init__()

        # Define the layers for multi-label classification of 2-grams
        layers = [
            nn.Linear(input_dim, hidden_layer),  # Input layer to first hidden layer
            activation_functions[activation_fn],  # Activation function
            nn.Dropout(dropout_rate),
        ]

        for _ in range(num_layers - 1):  # Add dynamic hidden layers
            layers.append(nn.Linear(hidden_layer, hidden_layer))  # Hidden layer to next hidden layer
            layers.append(activation_functions[activation_fn])  # Activation function
            layers.append(nn.Dropout(dropout_rate))  # Dropout layer

        layers.append(nn.Linear(hidden_layer, output_dim))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the model
        output = self.model(x)
        return output