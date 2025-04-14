import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer=256, num_layers=4, dropout_rate=0.2):
        super(TestModel, self).__init__()

        # Define the layers for multi-label classification of 2-grams
        layers = [
            nn.Linear(input_dim, hidden_layer),  # Input layer to first hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout_rate),
        ]

        for _ in range(num_layers - 1):  # Add dynamic hidden layers
            layers.append(nn.Linear(hidden_layer, hidden_layer))  # Hidden layer to next hidden layer
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.Dropout(dropout_rate))  # Dropout layer

        layers.append(nn.Linear(hidden_layer, output_dim))  # Output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the model
        output = self.model(x)
        return output


# Best hyperparameters: {'num_layers': 4, 'hidden_layer_size': 256, 'dropout_rate': 0.4844135931365565, 'activation_fn': 'gelu', 'optimizer': 'SGD', 'loss_fn': 'MultiLabelSoftMarginLoss', 'lr': 1.5087644320080698e-05, 'epochs': 29}