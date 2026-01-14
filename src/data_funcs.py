import ssl
import certifi
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import List
from .config import DATA_DIR

# Fix SSL certificate issue for downloading datasets on MacOS
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


def get_mnist_datasets():
    """Load MNIST train and test datasets."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def get_balanced_initial_indices(
    dataset: Dataset, n_per_class: int = 2, num_classes: int = 10, seed: int = 42
) -> List[int]:
    """
    Get balanced initial labeled set with n_per_class samples from each class.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    initial_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        selected = rng.choice(class_indices, size=n_per_class, replace=False)
        initial_indices.extend(selected.tolist())

    return initial_indices


def get_validation_indices(
    dataset: Dataset,
    n_total: int = 100,
    exclude_indices: List[int] = None,
    seed: int = 42,
) -> List[int]:
    """
    Get validation set indices (balanced across classes).
    Can ignore certain indicies (e.g. initial training indicies)
    """
    rng = np.random.RandomState(seed + 1000)  # Different seed from initial
    targets = np.array(dataset.targets)
    exclude_set = set(exclude_indices) if exclude_indices else set()

    num_classes = 10
    n_per_class = n_total // num_classes
    val_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        available = [i for i in class_indices if i not in exclude_set]
        selected = rng.choice(available, size=n_per_class, replace=False)
        val_indices.extend(selected.tolist())

    return val_indices
