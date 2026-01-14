# %%
import json
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

RESULTS_BASE_DIR = Path(__file__).parent.parent / "results"

# Used to add presentable labels to the legend
LABEL_NAME_MAP = {
    "independent_blr_simclr_k_max": "K-Max Acquisition",
    "independent_blr_simclr_k_centroids": "K-Centroids Acquisition",
    "independent_blr_simclr_variance": "Top-K Acquisition",
    "independent_blr_simclr_random": "Random Acquisition",
    "independent_blr_simple_retrain_random": "BLR + Random",
    "independent_blr_simple_retrain_variance": "BLR + Variance",
    "mfvi_simple_retrain_random": "MFVI + Random",
    "mfvi_simple_retrain_variance": "MFVI + Variance",
    "bald": "BALD",
    "entropy": "Entropy",
    "random": "Random",
    "variation_ratios": "Variation Ratios",
    "mean_std": "Mean STD",
}


# %%
def load_acquisition_results(acquisition_name: str, phase: int = 1) -> list[dict]:
    """Load all experiment results for a given acquisition function."""
    results_dir = RESULTS_BASE_DIR / f"phase{phase}"
    results = []
    for path in results_dir.glob(f"{acquisition_name}_seed*.json"):
        with open(path) as f:
            results.append(json.load(f))
    return results


def compute_average_metric(
    acquisition_name: str, phase: int = 1, metric: str = "test_acc"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average metric across all seeds for an acquisition function.
    """
    results = load_acquisition_results(acquisition_name, phase=phase)
    if not results:
        raise ValueError(
            f"No results found for acquisition function: {acquisition_name} in phase{phase}"
        )

    n_samples = np.array(results[0]["metrics"]["n_samples"])
    all_values = np.array([r["metrics"][metric] for r in results])

    mean_metric = all_values.mean(axis=0)
    std_metric = all_values.std(axis=0)

    return n_samples, mean_metric, std_metric


def compute_average_accuracy(
    acquisition_name: str, phase: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average accuracy across all seeds for an acquisition function.
    Convenience wrapper around compute_average_metric.
    """
    return compute_average_metric(acquisition_name, phase=phase, metric="test_acc")


def plot_all_runs(
    acquisition_name: str,
    phase: int = 1,
    metric: str = "test_acc",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot all individual runs for a single acquisition function.

    Args:
        acquisition_name: Name of the acquisition function
        phase: Phase number (1 or 2)
        metric: Metric to plot ("test_acc" or "test_rmse")
        y_min: Minimum y-axis value
        y_max: Maximum y-axis value
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    results = load_acquisition_results(acquisition_name, phase=phase)
    if not results:
        raise ValueError(
            f"No results found for acquisition function: {acquisition_name} in phase{phase}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    for r in results:
        n_samples = r["metrics"]["n_samples"]
        values = r["metrics"][metric]
        seed = r.get("seed", "?")
        ax.plot(n_samples, values, label=f"seed {seed}", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ylabel = "Test Accuracy" if metric == "test_acc" else "Test RMSE"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{acquisition_name} - All Runs (Phase {phase})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig


def plot_acquisition_comparison(
    acquisition_names: list[str],
    phase: int = 1,
    metric: str = "test_acc",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    show_std: bool = False,
    figsize: tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot metric vs training size for multiple acquisition functions. Can also add custome title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name in acquisition_names:
        n_samples, mean_val, std_val = compute_average_metric(
            name, phase=phase, metric=metric
        )

        line = ax.plot(
            n_samples, mean_val, label=LABEL_NAME_MAP.get(name, name), linewidth=2
        )[0]

        if show_std:
            ax.fill_between(
                n_samples,
                mean_val - std_val,
                mean_val + std_val,
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ylabel = "Test Accuracy" if metric == "test_acc" else "Test RMSE"
    ax.set_ylabel(ylabel, fontsize=12)
    if not (title):
        title = f"Acquisition Function Comparison (Phase {phase})"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig


# %%
methods = [
    "bald",
    "bald_deterministic",
    # "random", "entropy", "mean_std", "variation_ratios"
]
fig = plot_acquisition_comparison(
    methods, y_min=0.8, y_max=1.0, title="MC Acquisition Methods"
)
plt.show()
for name in methods:
    n_samples, mean_acc, std_acc = compute_average_metric(
        name, phase=1, metric="test_acc"
    )
    bound1 = False
    bound2 = False
    for i, samples in enumerate(n_samples):
        if mean_acc[i] > 0.9 and not (bound1):
            print(f"{name}_0.9 boundary: {samples}")
            bound1 = True
        if mean_acc[i] > 0.95 and not (bound2):
            bound2 = True
            print(f"{name}_0.95 boundary: {samples}")
        if i == len(n_samples) - 1:
            print(f"Final {name} accuracy: {mean_acc[i]}")

# %%
fig = plot_all_runs(
    "mfvi_simclr_variance",
    phase=2,
    metric="test_acc",
)
plt.show()
# %%
methods = [
    # "independent_blr_simple_retrain_variance",
    # "mfvi_simple_retrain_variance",
    # "independent_blr_simple_retrain_random",
    # "mfvi_simple_retrain_random",
    "independent_blr_simclr_variance",
    "independent_blr_simclr_random",
    "independent_blr_simclr_k_centroids",
    "independent_blr_simclr_k_max",
]
fig = plot_acquisition_comparison(
    methods,
    phase=2,
    metric="test_rmse",
    title="Independent BLR Inference with Clustered Acquisition (RMSE)",
)
plt.show()
for name in methods:
    n_samples, mean_acc, std_acc = compute_average_metric(
        name, phase=2, metric="test_acc"
    )
    bound1 = False
    bound2 = False
    n_samples, mean_rmse, std_acc = compute_average_metric(
        name, phase=2, metric="test_rmse"
    )
    for i, samples in enumerate(n_samples):
        if mean_acc[i] > 0.9 and not (bound1):
            print(f"{name}_0.9 boundary: {samples}")
            bound1 = True
        if mean_acc[i] > 0.95 and not (bound2):
            bound2 = True
            print(f"{name}_0.95 boundary: {samples}")
        if i == len(n_samples) - 1:
            print(f"{name} final rmse: {mean_rmse[i]}")
# %%
