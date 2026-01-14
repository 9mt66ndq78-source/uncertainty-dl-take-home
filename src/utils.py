import json
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JSONLogger:
    """Logger for saving experiment results to JSON."""

    def __init__(self, method: str, seed: int):
        self.data = {
            "method": method,
            "seed": seed,
            "metrics": {"n_samples": [], "test_acc": [], "test_nll": []},
            "acquisitions": [],
        }

    def log_cycle(self, n_samples: int, test_acc: float, test_nll: float):
        self.data["metrics"]["n_samples"].append(n_samples)
        self.data["metrics"]["test_acc"].append(test_acc)
        self.data["metrics"]["test_nll"].append(test_nll)

    def log_acquisition(self, cycle: int, indices: list, targets: list):
        """
        Log the indices and targets acquired this cycle.
        """
        self.data["acquisitions"].append(
            {"cycle": cycle, "indices": indices, "targets": targets}
        )

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)


class RegressionLogger:
    def __init__(self, method: str, seed: int):
        self.data = {
            "method": method,
            "seed": seed,
            "metrics": {
                "n_samples": [],
                "test_rmse": [],
                "test_acc": [],
            },
            "acquisitions": [],
        }

    def log(self, n_samples: int, rmse: float, acc: float):
        self.data["metrics"]["n_samples"].append(n_samples)
        self.data["metrics"]["test_rmse"].append(rmse)
        self.data["metrics"]["test_acc"].append(acc)

    def log_acquisition(self, cycle: int, indices: list, targets: list):
        """Log the indices and targets acquired this cycle."""
        self.data["acquisitions"].append(
            {
                "cycle": cycle,
                "indices": indices,
                "targets": targets,
            }
        )

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)


# turns seconds into minutes and hours
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
