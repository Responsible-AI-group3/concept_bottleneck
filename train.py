import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Standard and Joint training functions remain the same

def train_independent(concept_model, end_model, train_loader, cfg, device):
    # Train x → c
    c_optimizer = optim.SGD(concept_model.parameters(), lr=cfg.training.learning_rate, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
    c_criterion = nn.BCEWithLogitsLoss()
    
    c_losses = []
    for epoch in range(cfg.training.num_epochs):
        concept_model.train()
        epoch_loss = 0
        for inputs, concepts, _ in tqdm(train_loader, desc=f"Concept Training: Epoch {epoch+1}/{cfg.training.num_epochs}"):
            inputs, concepts = inputs.to(device), concepts.to(device)
            c_optimizer.zero_grad()
            c_outputs = concept_model(inputs)
            loss = c_criterion(c_outputs, concepts)
            loss.backward()
            c_optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        c_losses.append(avg_loss)
        print(f"Concept Training: Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Train c → y using true concepts
    y_optimizer = optim.SGD(end_model.parameters(), lr=cfg.training.learning_rate, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
    y_criterion = nn.CrossEntropyLoss()
    
    y_losses = []
    for epoch in range(cfg.training.num_epochs):
        end_model.train()
        epoch_loss = 0
        for _, concepts, labels in tqdm(train_loader, desc=f"End Model Training: Epoch {epoch+1}/{cfg.training.num_epochs}"):
            concepts, labels = concepts.to(device), labels.to(device)
            y_optimizer.zero_grad()
            y_outputs = end_model(concepts)
            loss = y_criterion(y_outputs, labels)
            loss.backward()
            y_optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        y_losses.append(avg_loss)
        print(f"End Model Training: Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return c_losses, y_losses

def train_sequential(concept_model, end_model, train_loader, cfg, device):
    # The concept model is already trained, so we only train c → y
    concept_model.eval()  # Ensure the concept model is in evaluation mode
    y_optimizer = optim.SGD(end_model.parameters(), lr=cfg.training.learning_rate, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
    y_criterion = nn.CrossEntropyLoss()
    
    y_losses = []
    for epoch in range(cfg.training.num_epochs):
        end_model.train()
        epoch_loss = 0
        for inputs, _, labels in tqdm(train_loader, desc=f"End Model Training: Epoch {epoch+1}/{cfg.training.num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            y_optimizer.zero_grad()
            with torch.no_grad():
                predicted_concepts = concept_model(inputs)
            y_outputs = end_model(predicted_concepts)
            loss = y_criterion(y_outputs, labels)
            loss.backward()
            y_optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        y_losses.append(avg_loss)
        print(f"End Model Training: Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return [], y_losses  # Return empty list for c_losses as concept model is not trained here

