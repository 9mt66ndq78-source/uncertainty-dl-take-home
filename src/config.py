# GLOBAL CONFIGS AND CONSTANTS
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()

# Phase 1 Hyperparameters, taken from the Gal et al. paper
PHASE1_CONFIG = {
    "initial_samples": 20,
    "samples_per_class": 2,
    "val_size": 100,
    "acquisition_steps": 100,
    "batch_size": 10,
    "mc_samples": 100,  # T for Monte Carlo dropout (evaluation)
    "mc_samples_acq": 20,  # T for acquisition, lower for performance reasons
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay_grid": [1e-3, 5e-4, 1e-4],  # Grid for tuning
    "retune_every": 10,  # How many cycles before tuning weight decay again
}

# Phase 2 Hyperparameters (the regression experiments)
PHASE2_CONFIG = {
    "pretrain_epochs": 30,  # Epochs for pre-training
    "feature_dim": 128,  # Feature dimension from CNN penultimate layer
    "initial_samples": 20,
    "acquisition_steps": 100,
    "batch_size": 10,  # Samples acquired per step
    # MFVI training
    "mfvi_epochs": 1000,  # Training epochs for MFVI
    "mfvi_lr": 0.01,  # Learning rate for MFVI
    # Bayesian parameters
    "prior_var": 1.0,  # Prior variance for weights
    "noise_var": 1.0,  # Observation noise variance
    # SimCLR pretraining parameters
    "simclr_epochs": 50,  # Epochs for SimCLR pretraining
    "simclr_lr": 0.001,  # Learning rate for SimCLR
    "simclr_batch_size": 256,
    "simclr_temperature": 0.5,  # Temperature for NT-Xent loss
    "simclr_projection_dim": 64,  # Projection head output dimension
    # Clustering acquisition parameters
    "top_n_multiplier": 5,
}
