"""
Modified version of the original code: https://github.com/yewsiang/ConceptBottleneck


"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

#from inception_model import MLP, inception_v3, End2EndModel
from  torchvision.models import inception_v3


# Independent & Sequential Model
class ModelXtoC(nn.Module):
    def __init__(self,pretrained, freeze,use_aux, n_attributes):
        """
        Model used for the X -> C part of the Independent and Sequential models

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            freeze (bool): If True, freezes the weights of the model except for the last layer
        
        Returns:
            model (nn.Module): Inception v3 model
        """
        super(ModelXtoC, self).__init__()

        self.use_aux = use_aux
        
        num_classes = n_attributes # number of classes is the number of attributes

        self.model = inception_v3(pretrained=pretrained, aux_logits=use_aux) # Load the inception model

        self.model.fc = nn.Linear(2048, num_classes) # Change the last layer to output the number of attributes

        self.activation = nn.Sigmoid() # Use sigmoid activation function


        if freeze:  # only finetune fc layer
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False

    def forward(self, x):
        Chat, aux_Chat = self.model(x)
        Chat = self.activation(Chat)

        if self.use_aux:
            aux_Chat = self.activation(aux_Chat)
            return Chat, aux_Chat
        
        return Chat


class ModelCtoY(nn.Module):
    """
    Simple one-layer MLP model for the C -> Y
    """
    def __init__(self, input_dim, num_classes):
        super(ModelCtoY, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x



# Joint Model
class ModelXtoCtoY(nn.Module):
    

    def __init__(self, pretrained, freeze, num_classes, use_aux, n_attributes):
        super(ModelXtoCtoY, self).__init__()

        self.sailency_output = False

        self.use_aux = use_aux

        self.CNN_model = inception_v3(pretrained=pretrained, aux_logits=use_aux) # Load the inception model

        self.CNN_model.fc = nn.Linear(2048, n_attributes) # Change the last layer to output the number of attributes

        self.activation = nn.Sigmoid() # Use sigmoid activation function

        self.MLP_model = ModelCtoY(n_attributes, num_classes) # Create the MLP model for the C -> Y

        if use_aux: #Make and auxilary model for the C -> Y
            self.CNN_model.AuxLogits.fc = nn.Linear(768, n_attributes)
            self.aux_MLP_model = ModelCtoY(n_attributes, num_classes)
        
        if freeze:  # only finetune fc layer
            for name, param in self.CNN_model():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the model
        """

        Chat, aux_Chat = self.CNN_model(x)
        Chat = self.activation(Chat)
        Yhat = self.MLP_model(Chat)

        if self.use_aux and self.training: 
            aux_Chat = self.activation(aux_Chat)
            aux_Yhat = self.aux_MLP_model(aux_Chat)
            return Chat, Yhat, aux_Chat, aux_Yhat
        else:
            if not self.sailency_output:
                return Chat, Yhat

            else:
                # Only return the output we want to create the sailency map of
                if self.sailency_output == 'C':
                    return Chat
                elif self.sailency_output == 'Y':
                    return Yhat 
    
    def set_sailency_output(self, output):
        """
        Function for returning the sailency map of the model
        args:
            output (str): Either 'C' or 'Y' for the sailency map of the Chat or Yhat
        """

        if output in ['C', 'Y']:
            self.sailency_output = output
        else:
            raise ValueError("Invalid output type for sailency map. Please provide one of the following: C, Y")
        


# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    model = inception_v3(pretrained=pretrained, aux_logits=use_aux) # Load the inception model

    model.fc = nn.Linear(2048, num_classes) # Change the last layer to output the number of attributes

    if freeze:  # only finetune fc layer
        for name, param in model():
            if 'fc' not in name:  # and 'Mixed_7c' not in name:
                param.requires_grad = False
    return model



