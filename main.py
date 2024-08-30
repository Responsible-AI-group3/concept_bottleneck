import torch
from torch.utils.data import DataLoader
from ConceptModel import get_concept_model
from EndModel import get_end_classifier
from dataloader import get_dataset
from train import train_standard, train_independent, train_sequential, train_joint
from utils.ploting import plot_results
from utils.model_utils import save_models, load_concept_model
import os
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def get_dataloaders(cfg):
    train_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=True, majority_voting=cfg.dataset.majority_voting)
    val_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=False, majority_voting=cfg.dataset.majority_voting)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_device(cfg):
    if cfg.device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration for this run:")
    print(OmegaConf.to_yaml(cfg))
    
    device = get_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    
    output_dir = Path(HydraConfig.get().run.dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    for mode in cfg.modes:
        print(f"\nRunning experiment with mode: {mode}")
        
        concept_model = get_concept_model(cfg.models.concept_model, cfg.models.num_concepts, cfg.models.pretrained).to(device)
        end_model = get_end_classifier(cfg.models.end_model_layers).to(device)
        
        if mode == 'standard':
            results = train_standard(concept_model, end_model, train_loader, val_loader, cfg.training, device)
        elif mode == 'independent':
            results = train_independent(concept_model, end_model, train_loader, val_loader, cfg.training, device)
        elif mode == 'sequential':
            loaded_concept_model = load_concept_model(concept_model, 'independent', models_dir)
            if loaded_concept_model is None:
                print("Pre-trained concept model not found. Please run independent mode first.")
                continue
            concept_model = loaded_concept_model
            results = train_sequential(concept_model, end_model, train_loader, val_loader, cfg.training, device)
        elif mode == 'joint':
            results = train_joint(concept_model, end_model, train_loader, val_loader, cfg.training, device)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        save_models(concept_model, end_model, mode, models_dir)
        plot_results(results, cfg, mode, plots_dir)
        
        print(f"Experiment for mode {mode} completed!")
    
    print("\nAll experiments completed!")
    print(f"Output files are saved in: {output_dir}")

if __name__ == "__main__":
    main()