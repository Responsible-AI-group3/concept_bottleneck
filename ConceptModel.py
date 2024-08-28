import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class ResNetModel(nn.Module):
    def __init__(self, num_concepts: int = 312, pretrained: bool = True):
        """
        Initialize the ResNet model for concept prediction.

        Args:
            num_concepts (int): Number of concepts to predict.
            pretrained (bool): Whether to use pretrained weights or random initialization.
        """
        super(ResNetModel, self).__init__()
        
        # Initialize ResNet18 model
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Predicted concept probabilities.
        """
        x = self.resnet(x)
        return torch.sigmoid(x)

class InceptionModel(nn.Module):
    def __init__(self, num_concepts: int = 312, pretrained: bool = True):
        """
        Initialize the Inception v3 model for concept prediction.

        Args:
            num_concepts (int): Number of concepts to predict.
            pretrained (bool): Whether to use pretrained weights or random initialization.
        """
        super(InceptionModel, self).__init__()
        
        # Initialize Inception v3 model
        if pretrained:
            self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        else:
            self.inception = models.inception_v3(weights=None)

        # Replace the final fully connected layer
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, num_concepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception v3 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Predicted concept probabilities.
        """
        # Inception v3 returns tuple in training mode
        if self.inception.training:
            x, _ = self.inception(x)
        else:
            x = self.inception(x)
        return torch.sigmoid(x)

def get_concept_model(model_name: str, num_concepts: int = 312, pretrained: bool = True) -> Optional[nn.Module]:
    """
    Factory function to get a concept prediction model.

    Args:
        model_name (str): Name of the model ('resnet' or 'inception').
        num_concepts (int): Number of concepts to predict.
        pretrained (bool): Whether to use pretrained weights or random initialization.

    Returns:
        nn.Module or None: The requested model, or None if the model name is invalid.
    """
    if model_name.lower() == 'resnet':
        return ResNetModel(num_concepts, pretrained)
    elif model_name.lower() == 'inception':
        return InceptionModel(num_concepts, pretrained)
    else:
        print(f"Invalid model name: {model_name}. Choose 'resnet' or 'inception'.")
        return None