import matplotlib.pyplot as plt
from omegaconf import DictConfig
import os

def plot_losses(c_losses, y_losses, cfg: DictConfig, mode: str):
    """
    Plot and save the training losses.

    Args:
        c_losses (list): Concept losses over epochs.
        y_losses (list): Classification losses over epochs.
        cfg (DictConfig): Configuration object.
        mode (str): The training mode used (e.g., 'independent', 'joint').
    """
    plt.figure(figsize=(10, 5))
    plt.plot(c_losses, label='Concept Loss')
    plt.plot(y_losses, label='Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{mode.capitalize()} Training Losses')
    plt.legend()
    
    plt.savefig(os.path.join("plots", f"losses_{mode}.png"))
    plt.close()