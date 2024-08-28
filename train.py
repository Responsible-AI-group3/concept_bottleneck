import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train(
    concept_model: nn.Module,
    end_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    mode: str,
    num_epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    lambda1: float,
    device: torch.device,
    verbose: bool = True
):
    """
    Train the concept and end models based on the specified mode.
    
    Args:
        concept_model (nn.Module): The concept prediction model
        end_model (nn.Module): The end classifier model
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        mode (str): Training mode ('standard', 'independent', 'joint', or 'sequential')
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        momentum (float): Momentum for SGD optimizer
        weight_decay (float): Weight decay for regularization
        lambda1 (float): Lambda parameter for joint training
        device (torch.device): Device to run the training on
        verbose (bool): Whether to print detailed progress
    
    Returns:
        tuple: Lists of concept and classification losses over epochs
    """
    criterion = nn.CrossEntropyLoss()
    
    if mode in ['standard', 'joint']:
        params = list(concept_model.parameters()) + list(end_model.parameters())
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif mode == 'independent':
        concept_optimizer = optim.SGD(concept_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        end_optimizer = optim.SGD(end_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif mode == 'sequential':
        optimizer = optim.SGD(end_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    c_losses, y_losses = [], []
    
    for epoch in tqdm(range(num_epochs), desc='Training', disable=not verbose):
        concept_model.train()
        end_model.train()
        c_loss_sum, y_loss_sum = 0, 0
        
        for X, c, y in train_loader:
            X, c, y = X.to(device), c.to(device), y.to(device)
            
            if mode in ['standard', 'joint', 'sequential']:
                optimizer.zero_grad()
            elif mode == 'independent':
                concept_optimizer.zero_grad()
                end_optimizer.zero_grad()
            
            c_pred = concept_model(X)
            y_pred = end_model(c_pred)
            
            if mode == 'standard':
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                y_loss_sum += loss.item()
            elif mode == 'independent':
                c_loss = criterion(c_pred, c)
                c_loss.backward()
                concept_optimizer.step()
                
                y_loss = criterion(y_pred, y)
                y_loss.backward()
                end_optimizer.step()
                
                c_loss_sum += c_loss.item()
                y_loss_sum += y_loss.item()
            elif mode == 'joint':
                c_loss = criterion(c_pred, c)
                y_loss = criterion(y_pred, y)
                loss = y_loss + lambda1 * c_loss
                loss.backward()
                optimizer.step()
                
                c_loss_sum += c_loss.item()
                y_loss_sum += y_loss.item()
            elif mode == 'sequential':
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                y_loss_sum += loss.item()
        
        c_loss_avg = c_loss_sum / len(train_loader) if c_loss_sum > 0 else 0
        y_loss_avg = y_loss_sum / len(train_loader)
        c_losses.append(c_loss_avg)
        y_losses.append(y_loss_avg)
        
        if verbose:
            print(f'Epoch: {epoch + 1}, Concept Loss: {c_loss_avg:.4f}, Classification Loss: {y_loss_avg:.4f}')
    
    return c_losses, y_losses

def save_models(concept_model, end_model, mode):
    """
    Save the trained models.

    Args:
        concept_model (nn.Module): The trained concept model.
        end_model (nn.Module): The trained end model.
        mode (str): The training mode used (e.g., 'independent', 'joint').
    """
    torch.save(concept_model.state_dict(), os.path.join("models", f"concept_model_{mode}.pth"))
    torch.save(end_model.state_dict(), os.path.join("models", f"end_model_{mode}.pth"))
