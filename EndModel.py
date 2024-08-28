import torch
import torch.nn as nn
from typing import List, Optional

class EndClassifier(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: Optional[str] = 'relu'):
        """
        Initialize the End Classifier model.

        Args:
            layer_sizes (List[int]): A list of integers representing the size of each layer.
                                     The first element is the input size, and the last is the output size.
            activation (Optional[str]): The activation function to use between layers. 
                                        Options: 'relu', 'tanh', or None. Default is 'relu'.
        """
        super(EndClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the End Classifier model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation after the last layer
                x = self.activation(x)
        return x


def get_end_classifier(layer_sizes: List[int], activation: Optional[str] = 'relu') -> EndClassifier:
    """
    Factory function to get an End Classifier model.

    Args:
        layer_sizes (List[int]): A list of integers representing the size of each layer.
                                 The first element is the input size, and the last is the output size.
        activation (Optional[str]): The activation function to use between layers. 
                                    Options: 'relu', 'tanh', or None. Default is 'relu'.

    Returns:
        EndClassifier: The requested End Classifier model.
    """
    return EndClassifier(layer_sizes, activation)