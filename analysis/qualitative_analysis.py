# %%
"""Qualitative analysis of active learning acquisitions."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# %%
# Configuration - set the path to your results file
RESULTS_FILE = (
    Path(__file__).parent.parent
    / "results/phase2/independent_blr_simclr_variance_seed1.json"
)
DATA_DIR = Path(__file__).parent.parent / "data"

# MNIST class names (digits 0-9)
CLASS_NAMES = [str(i) for i in range(10)]


# %%
def load_results(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def load_mnist_train():
    transform = transforms.ToTensor()
    return datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)


# %%
def plot_acquired_images_at_cycle(
    dataset,
    acquisitions: list[dict],
    cycle: int,
    figsize: tuple[int, int] = (12, 3),
) -> plt.Figure:
    """
    Plot images acquired at a specific cycle.
    """
    acq = next((a for a in acquisitions if a["cycle"] == cycle), None)
    if acq is None:
        raise ValueError(f"Cycle {cycle} not found in acquisitions")

    indices = acq["indices"]
    targets = acq["targets"]
    n_samples = len(indices)

    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    if n_samples == 1:
        axes = [axes]

    for ax, idx, target in zip(axes, indices, targets):
        img, _ = dataset[idx]
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[target]}", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Cycle {cycle} Acquisitions", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def compare_acquisitions_at_cycle(
    dataset,
    results_path_1: Path,
    results_path_2: Path,
    name_1: str,
    name_2: str,
    cycle: int,
    figsize: tuple[int, int] = (12, 3),
) -> plt.Figure:
    """
    Compare acquired images from two methods at a specific cycle.
    """
    results_1 = load_results(results_path_1)
    results_2 = load_results(results_path_2)

    acq_1 = next((a for a in results_1["acquisitions"] if a["cycle"] == cycle), None)
    acq_2 = next((a for a in results_2["acquisitions"] if a["cycle"] == cycle), None)

    if acq_1 is None:
        raise ValueError(f"Cycle {cycle} not found in {results_path_1}")
    if acq_2 is None:
        raise ValueError(f"Cycle {cycle} not found in {results_path_2}")

    n_samples = max(len(acq_1["indices"]), len(acq_2["indices"]))

    fig = plt.figure(figsize=(figsize[0], figsize[1] * 2))
    gs = fig.add_gridspec(
        2,
        n_samples + 1,
        width_ratios=[0.8] + [1] * n_samples,
        wspace=0.05,
        hspace=0.3,
    )

    for row_idx, (acq, name) in enumerate([(acq_1, name_1), (acq_2, name_2)]):
        # Add method label in first column
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.text(
            0.5,
            0.5,
            name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.axis("off")

        # Add images in remaining columns
        indices = acq["indices"]
        targets = acq["targets"]
        for col_idx, (idx, target) in enumerate(zip(indices, targets)):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            img, _ = dataset[idx]
            ax.imshow(img.squeeze().numpy(), cmap="gray")
            ax.set_title(f"{CLASS_NAMES[target]}", fontsize=9)
            ax.axis("off")

    fig.suptitle(f"Cycle {cycle} Acquisitions", fontsize=14)
    plt.tight_layout()
    return fig


def plot_acquired_images_every_n_cycles(
    dataset,
    acquisitions: list[dict],
    every_n: int = 10,
    samples_per_row: int = 10,
) -> plt.Figure:
    """
    Plot acquired images at every N cycles in a grid with clear cycle labels.
    """
    cycles_to_show = [a["cycle"] for a in acquisitions if a["cycle"] % every_n == 0]
    n_rows = len(cycles_to_show)

    # Create figure with extra space on the left for cycle labels
    fig = plt.figure(figsize=(samples_per_row * 1.2 + 1.5, n_rows * 1.5))

    # Use gridspec to create layout with label column
    gs = fig.add_gridspec(
        n_rows,
        samples_per_row + 1,
        width_ratios=[0.8] + [1] * samples_per_row,
        wspace=0.05,
        hspace=0.3,
    )

    for row_idx, cycle in enumerate(cycles_to_show):
        acq = next(a for a in acquisitions if a["cycle"] == cycle)
        indices = acq["indices"][:samples_per_row]
        targets = acq["targets"][:samples_per_row]

        # Add cycle label in first column
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.text(
            0.5,
            0.5,
            f"Cycle\n{cycle}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.axis("off")

        # Add images in remaining columns
        for col_idx, (idx, target) in enumerate(zip(indices, targets)):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            img, _ = dataset[idx]
            ax.imshow(img.squeeze().numpy(), cmap="gray")
            ax.set_title(f"{target}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Acquired Images Every 10 Cycles", fontsize=14)
    return fig


# %%
results = load_results(RESULTS_FILE)
mnist_train = load_mnist_train()

# %%
fig = plot_acquired_images_every_n_cycles(
    mnist_train,
    results["acquisitions"],
    every_n=10,
)
plt.show()

# %%
fig = plot_acquired_images_at_cycle(mnist_train, results["acquisitions"], cycle=5)
plt.show()

# %%
file1 = (
    Path(__file__).parent.parent
    / "results/phase2/independent_blr_simclr_variance_seed1.json"
)
file2 = (
    Path(__file__).parent.parent
    / "results/phase2/independent_blr_simclr_k_centroids_seed1.json"
)
compare_acquisitions_at_cycle(
    mnist_train, file1, file2, "Top-K", "K-Centroids", 5, (16, 2)
)
# %%
