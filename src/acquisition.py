import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Tuple
import time

# Small constant for numerical stability
EPS = 1e-10


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy: -sum(p * log(p))."""
    return -(probs * torch.log(probs + EPS)).sum(dim=-1)


def random_acquisition(
    pool_size: int, batch_size: int, rng: np.random.RandomState
) -> List[int]:
    """Random acquisition baseline."""
    return rng.choice(pool_size, size=batch_size, replace=False).tolist()


def acquire_batch(
    model,
    pool_loader: DataLoader,
    batch_size: int,
    method: str,
    mc_samples: int,
    device: torch.device,
    rng: np.random.RandomState = None,
    deterministic: bool = False,
) -> Tuple[List[int], float]:
    """
    Acquire a batch of samples using the specified method.
    """
    start_time = time.time()

    if method == "random":
        pool_size = len(pool_loader.dataset)
        selected = random_acquisition(pool_size, batch_size, rng)
        return selected, time.time() - start_time

    needs_samples = method in ["bald", "mean_std"]

    all_scores = []

    pbar = tqdm(pool_loader, desc=f"Acquisition ({method})", leave=False)
    for batch_x, _ in pbar:
        batch_x = batch_x.to(device)

        if deterministic:
            # Single forward pass with dropout disabled
            model.eval()
            with torch.no_grad():
                log_probs = model(batch_x, apply_dropout=False)
                mean_probs = torch.exp(log_probs)
                # Shape 1xBxC so existing code works
                samples = mean_probs.unsqueeze(0) if needs_samples else None
        else:
            mean_probs, samples = model.predict_proba(
                batch_x, n_samples=mc_samples, return_samples=needs_samples
            )

        if method == "bald":
            predictive_entropy = compute_entropy(mean_probs)  # [B]
            sample_entropies = compute_entropy(samples)  # [T, B]
            expected_entropy = sample_entropies.mean(dim=0)  # [B]
            batch_scores = predictive_entropy - expected_entropy
        elif method == "entropy":
            batch_scores = compute_entropy(mean_probs)
        elif method == "variation_ratios":
            batch_scores = 1.0 - mean_probs.max(dim=1)[0]
        elif method == "mean_std":
            std_per_class = samples.std(dim=0)  # [B, C]
            batch_scores = std_per_class.mean(dim=1)  # [B]
        else:
            raise ValueError(f"Unknown acquisition method: {method}")

        all_scores.append(batch_scores.cpu())

    # Concatenate all scores and select top-k
    all_scores = torch.cat(all_scores, dim=0)
    _, top_indices = torch.topk(all_scores, k=batch_size)
    selected = top_indices.tolist()

    return selected, time.time() - start_time


def cluster_and_select(
    features: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    method: str = "k_max",
) -> List[int]:
    """
    Implements k_max and k_centroid selection
    """

    # Need to move to CPU for sklearn
    features_np = features.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_np)

    selected_indices = []

    if method == "k_centroids":
        # Select sample closest to each centroid
        centroids = kmeans.cluster_centers_
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find sample closest to centroid
            cluster_features = features_np[cluster_indices]
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

    else:  # k_max
        # Select highest-scoring sample from each cluster
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue
            cluster_scores = scores_np[cluster_indices]
            best_idx = cluster_indices[np.argmax(cluster_scores)]
            selected_indices.append(best_idx)

    return selected_indices
