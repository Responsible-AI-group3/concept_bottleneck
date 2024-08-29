import torch
from torch.utils.data import DataLoader
from ConceptModel import get_concept_model
from EndModel import get_end_classifier
from dataloader import get_dataset
from train import train_standard, train_independent, train_sequential, train_joint
from utils.ploting import plot_losses
from utils.model_utils import save_models, load_concept_model, load_end_model
import os
import hydra
from omegaconf import DictConfig, OmegaConf

def get_dataloaders(cfg):
    """
    Create and return data loaders for training and testing.

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters.

    Returns:
        tuple: (train_loader, test_loader) - DataLoader objects for training and testing.
    """
    train_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=True, majority_voting=cfg.dataset.majority_voting)
    test_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=False, majority_voting=cfg.dataset.majority_voting)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_device(cfg):
    """
    Determine the device to use for training.

    Args:
        cfg (DictConfig): Configuration object containing the device setting.

    Returns:
        torch.device: The device to use for training.
    """
    if cfg.device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration for this run:")
    print(OmegaConf.to_yaml(cfg))
    
    device = get_device(cfg)
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(cfg)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    for mode in cfg.modes:
        print(f"\nRunning experiment with mode: {mode}")
        
        concept_model = get_concept_model(cfg.models.concept_model, cfg.models.num_concepts, cfg.models.pretrained).to(device)
        end_model = get_end_classifier(cfg.models.end_model_layers).to(device)
        
        if mode == 'standard':
            losses = train_standard(concept_model, end_model, train_loader, test_loader, mode, cfg.training.num_epochs, 
                                    cfg.training.learning_rate, cfg.training.momentum, cfg.training.weight_decay, 
                                    cfg.training.lambda1, device, verbose=True)
            c_losses, y_losses = losses, losses
        elif mode == 'independent':
            c_losses, y_losses = train_independent(concept_model, end_model, train_loader, test_loader, mode, 
                                                   cfg.training.num_epochs, cfg.training.learning_rate, cfg.training.momentum, 
                                                   cfg.training.weight_decay, cfg.training.lambda1, device, verbose=True)
        elif mode == 'sequential':
            loaded_concept_model = load_concept_model(concept_model, 'independent')
            if loaded_concept_model is None:
                print("Pre-trained concept model not found. Please run independent mode first.")
                continue  # Skip this mode if the pre-trained model is not available
            concept_model = loaded_concept_model
            c_losses, y_losses = train_sequential(concept_model, end_model, train_loader, test_loader, mode, 
                                                  cfg.training.num_epochs, cfg.training.learning_rate, cfg.training.momentum, 
                                                  cfg.training.weight_decay, cfg.training.lambda1, device, verbose=True)
        elif mode == 'joint':
            c_losses, y_losses = train_joint(concept_model, end_model, train_loader, test_loader, mode, 
                                             cfg.training.num_epochs, cfg.training.learning_rate, cfg.training.momentum, 
                                             cfg.training.weight_decay, cfg.training.lambda1, device, verbose=True)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Save models
        save_models(concept_model, end_model, mode)
        
        # Plot losses
        plot_losses(c_losses, y_losses, cfg, mode)
        
        print(f"Experiment for mode {mode} completed!")
    
    print("\nAll experiments completed!")
    print(f"Output files are saved in: {os.getcwd()}")

if __name__ == "__main__":
    main()