import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

def train_standard(concept_model, end_model, train_loader, val_loader, cfg, device, verbose=True):
    combined_model = nn.Sequential(concept_model, end_model)
    optimizer = optim.SGD(combined_model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(cfg.num_epochs):
        combined_model.train()
        train_loss = 0.0
        for inputs, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", disable=not verbose):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = combined_model(inputs)
            loss = criterion(outputs, labels.argmax(1))  # Use argmax for multi-class
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        combined_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, _, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = combined_model(inputs)
                loss = criterion(outputs, labels.argmax(1))  # Use argmax for multi-class
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(1)).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        results = {
        'train_losses': {'class': train_losses},
        'val_losses': {'class': val_losses},
        'val_accuracies': {'class': val_accuracies}
    }
    return results

def train_independent(concept_model, end_model, train_loader, val_loader, cfg, device, verbose=True):
    concept_optimizer = optim.SGD(concept_model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    end_optimizer = optim.SGD(end_model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    concept_scheduler = StepLR(concept_optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    end_scheduler = StepLR(end_optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    concept_criterion = nn.BCEWithLogitsLoss()
    class_criterion = nn.CrossEntropyLoss()

    train_losses = {'concept': [], 'class': []}
    val_losses = {'concept': [], 'class': []}
    val_accuracies = {'concept': [], 'class': []}

    for epoch in range(cfg.num_epochs):
        concept_model.train()
        end_model.train()
        concept_train_loss = 0.0
        class_train_loss = 0.0

        for inputs, concepts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", disable=not verbose):
            inputs, concepts, labels = inputs.to(device), concepts.to(device), labels.to(device)
            
            # Train concept model
            concept_optimizer.zero_grad()
            concept_outputs = concept_model(inputs)
            concept_loss = concept_criterion(concept_outputs, concepts)
            concept_loss.backward()
            concept_optimizer.step()
            concept_train_loss += concept_loss.item()

            # Train end model
            end_optimizer.zero_grad()
            class_outputs = end_model(concepts)
            class_loss = class_criterion(class_outputs, labels.argmax(1))  # Use argmax for multi-class
            class_loss.backward()
            end_optimizer.step()
            class_train_loss += class_loss.item()

        concept_train_loss /= len(train_loader)
        class_train_loss /= len(train_loader)
        train_losses['concept'].append(concept_train_loss)
        train_losses['class'].append(class_train_loss)

        concept_model.eval()
        end_model.eval()
        concept_val_loss = 0.0
        class_val_loss = 0.0
        concept_correct = 0
        class_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, concepts, labels in val_loader:
                inputs, concepts, labels = inputs.to(device), concepts.to(device), labels.to(device)
                concept_outputs = concept_model(inputs)
                class_outputs = end_model(concepts)

                concept_loss = concept_criterion(concept_outputs, concepts)
                class_loss = class_criterion(class_outputs, labels.argmax(1))  # Use argmax for multi-class

                concept_val_loss += concept_loss.item()
                class_val_loss += class_loss.item()

                concept_correct += ((concept_outputs > 0.5) == concepts).sum().item()
                _, predicted = class_outputs.max(1)
                class_correct += (predicted == labels.argmax(1)).sum().item()
                total += labels.size(0)

        concept_val_loss /= len(val_loader)
        class_val_loss /= len(val_loader)
        concept_accuracy = concept_correct / (total * concepts.size(1))
        class_accuracy = class_correct / total

        val_losses['concept'].append(concept_val_loss)
        val_losses['class'].append(class_val_loss)
        val_accuracies['concept'].append(concept_accuracy)
        val_accuracies['class'].append(class_accuracy)

        concept_scheduler.step()
        end_scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{cfg.num_epochs}")
            print(f"Concept - Train Loss: {concept_train_loss:.4f}, Val Loss: {concept_val_loss:.4f}, Val Acc: {concept_accuracy:.4f}")
            print(f"Class - Train Loss: {class_train_loss:.4f}, Val Loss: {class_val_loss:.4f}, Val Acc: {class_accuracy:.4f}")

        results = {
        'train_losses': {'concept': concept_train_loss, 'class': class_train_loss},
        'val_losses': {'concept': concept_val_loss, 'class': class_val_loss},
        'val_accuracies': {'concept': class_accuracy, 'class': concept_accuracy}
    }
    return results

def train_sequential(concept_model, end_model, train_loader, val_loader, cfg, device, verbose=True):
    optimizer = optim.SGD(end_model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(cfg.num_epochs):
        end_model.train()
        train_loss = 0.0

        for inputs, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", disable=not verbose):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                concepts = concept_model(inputs)
            outputs = end_model(concepts)
            loss = criterion(outputs, labels.argmax(1))  # Use argmax for multi-class
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        end_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, _, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                concepts = concept_model(inputs)
                outputs = end_model(concepts)
                loss = criterion(outputs, labels.argmax(1))  # Use argmax for multi-class
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(1)).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
    results = {
        'train_losses': {'class': train_losses},
        'val_losses': {'class': val_losses},
        'val_accuracies': {'class': val_accuracies}
    }
    return results

def train_joint(concept_model, end_model, train_loader, val_loader, cfg, device, verbose=True):
    optimizer = optim.SGD(list(concept_model.parameters()) + list(end_model.parameters()), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_scheduler.step_size, gamma=cfg.lr_scheduler.gamma)
    concept_criterion = nn.BCEWithLogitsLoss()
    class_criterion = nn.CrossEntropyLoss()

    train_losses = {'concept': [], 'class': []}
    val_losses = {'concept': [], 'class': []}
    val_accuracies = {'concept': [], 'class': []}

    for epoch in range(cfg.num_epochs):
        concept_model.train()
        end_model.train()
        concept_train_loss = 0.0
        class_train_loss = 0.0

        for inputs, concepts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", disable=not verbose):
            inputs, concepts, labels = inputs.to(device), concepts.to(device), labels.to(device)
            optimizer.zero_grad()
            concept_outputs = concept_model(inputs)
            class_outputs = end_model(concept_outputs)
            concept_loss = concept_criterion(concept_outputs, concepts)
            class_loss = class_criterion(class_outputs, labels.argmax(1))  # Use argmax for multi-class
            loss = class_loss + cfg.lambda1 * concept_loss
            loss.backward()
            optimizer.step()
            concept_train_loss += concept_loss.item()
            class_train_loss += class_loss.item()

        concept_train_loss /= len(train_loader)
        class_train_loss /= len(train_loader)
        train_losses['concept'].append(concept_train_loss)
        train_losses['class'].append(class_train_loss)

        concept_model.eval()
        end_model.eval()
        concept_val_loss = 0.0
        class_val_loss = 0.0
        concept_correct = 0
        class_correct = 0
        total = 0

        with torch.no_grad():
            for inputs, concepts, labels in val_loader:
                inputs, concepts, labels = inputs.to(device), concepts.to(device), labels.to(device)
                concept_outputs = concept_model(inputs)
                class_outputs = end_model(concept_outputs)
                concept_loss = concept_criterion(concept_outputs, concepts)
                class_loss = class_criterion(class_outputs, labels.argmax(1))  # Use argmax for multi-class
                concept_val_loss += concept_loss.item()
                class_val_loss += class_loss.item()
                concept_correct += ((concept_outputs > 0.5) == concepts).sum().item()
                _, predicted = class_outputs.max(1)
                class_correct += (predicted == labels.argmax(1)).sum().item()
                total += labels.size(0)

        concept_val_loss /= len(val_loader)
        class_val_loss /= len(val_loader)
        concept_accuracy = concept_correct / (total * concepts.size(1))
        class_accuracy = class_correct / total

        val_losses['concept'].append(concept_val_loss)
        val_losses['class'].append(class_val_loss)
        val_accuracies['concept'].append(concept_accuracy)
        val_accuracies['class'].append(class_accuracy)

        scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{cfg.num_epochs}")
            print(f"Train - Concept Loss: {concept_train_loss:.4f}, Class Loss: {class_train_loss:.4f}")
            print(f"Val - Concept Loss: {concept_val_loss:.4f}, Class Loss: {class_val_loss:.4f}")
            print(f"Val - Concept Acc: {concept_accuracy:.4f}, Class Acc: {class_accuracy:.4f}")

        results = {
        'train_losses': {'concept': concept_train_loss, 'class': class_train_loss},
        'val_losses': {'concept': concept_val_loss, 'class': class_val_loss},
        'val_accuracies': {'concept': class_accuracy, 'class': concept_accuracy}
    }
    return results