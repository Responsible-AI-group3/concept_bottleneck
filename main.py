import torch
from torch.utils.data import DataLoader
from ConceptModel import get_concept_model
from EndModel import get_end_classifier

from utils.ploting import plot_results
from utils.model_utils import save_models, load_concept_model

import pickle
import os
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
from train import train_X_to_C,train_X_to_C_to_y,train_X_to_y,train_C_to_Y

"""
def get_dataloaders(cfg):
    train_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=True, majority_voting=cfg.dataset.majority_voting,cfg.dataset.threshold)
    val_dataset = get_dataset(cfg.dataset.name, cfg.dataset.path, train=False, majority_voting=cfg.dataset.majority_voting,cfg.dataset.threshold)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    return train_loader, val_loader
"""
def get_device(cfg):
    if cfg.device.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(cfg.device)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig):

    args.log_dir = Path(HydraConfig.get().run.dir) # put the log files in the same directory as the output

    
    # Find the device
    device = get_device(args)
    print(f"Using device: {device}")
    args.device = device

    experiment = args.mode
    

    if not isinstance(args.n_attributes, (int, float)): # if the data is dynamic, we need to load the data to get the dimensions
        data = pickle.load(open(os.path.join(args.data_dir,"train.pkl"), 'rb'))
        args.n_attributes = len(data[0]['attribute_label'])


    print("Configuration for this run:")
    print(OmegaConf.to_yaml(args))
    

    if experiment == 'Concept':
        train_X_to_C(args)

    elif experiment == 'Independent':
        train_X_to_C(args)

    elif experiment == 'Sequential':
        train_C_to_Y(args)

    elif experiment == 'Joint':
        args.use_attr = True
        train_X_to_C_to_y(args)

    elif experiment == 'Standard':
        train_X_to_y(args)
    
    else:
        print(f"Invalid experiment type {experiment} provided. Please provide one of the following: Concept, Independent, Sequential, Joint, Standard")


if __name__ == "__main__":
    main()