import torch.nn as nn

class BloomFilterToTwoGramClassifier(nn.Module):
    def __init__(self, input_dim, num_two_grams, hidden_layer=512):
        super(BloomFilterToTwoGramClassifier, self).__init__()

        # Define the layers for multi-label classification of 2-grams
        self.model = nn.Sequential(
        nn.Linear(input_dim, hidden_layer),  # Input => first hidden layer
        nn.ReLU(),
        #nn.Dropout(0.2),
        nn.Linear(hidden_layer, hidden_layer),  # First hidden layer => second hidden layer
        nn.ReLU(),
        #nn.Dropout(0.2),
        nn.Linear(hidden_layer, num_two_grams),  # Second hidden layer => output layer
        )

    def forward(self, x):
        # Forward pass through the model
        output = self.model(x)
        return output