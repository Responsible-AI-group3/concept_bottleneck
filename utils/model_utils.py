import os
import torch

def save_models(concept_model, end_model, mode, save_dir="models"):
    """
    Save both concept and end models.
    
    Args:
    - concept_model: The concept model to save
    - end_model: The end model to save
    - mode: The training mode (e.g., 'independent', 'joint')
    - save_dir: Directory to save the models
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(concept_model.state_dict(), os.path.join(save_dir, f"concept_model_{mode}.pth"))
    torch.save(end_model.state_dict(), os.path.join(save_dir, f"end_model_{mode}.pth"))
    print(f"Models saved for mode: {mode}")

def load_concept_model(concept_model, mode, load_dir="models"):
    """
    Load a concept model.
    
    Args:
    - concept_model: The concept model architecture to load weights into
    - mode: The training mode of the saved model
    - load_dir: Directory to load the model from
    
    Returns:
    - loaded_model: The loaded concept model
    """
    model_path = os.path.join(load_dir, f"concept_model_{mode}.pth")
    if os.path.exists(model_path):
        concept_model.load_state_dict(torch.load(model_path))
        print(f"Loaded concept model for mode: {mode}")
        return concept_model
    else:
        print(f"No saved concept model found for mode: {mode}")
        return None

def load_end_model(end_model, mode, load_dir="models"):
    """
    Load an end model.
    
    Args:
    - end_model: The end model architecture to load weights into
    - mode: The training mode of the saved model
    - load_dir: Directory to load the model from
    
    Returns:
    - loaded_model: The loaded end model
    """
    model_path = os.path.join(load_dir, f"end_model_{mode}.pth")
    if os.path.exists(model_path):
        end_model.load_state_dict(torch.load(model_path))
        print(f"Loaded end model for mode: {mode}")
        return end_model
    else:
        print(f"No saved end model found for mode: {mode}")
        return None