import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, format_time, RegressionLogger
from src.train_funcs import train_model, extract_features
from src.models import (
    BayesianCNN,
    IndependentBLR,
    MFVIRegression,
    MatrixNormalRegression,
)
from src.data_funcs import get_mnist_datasets, get_balanced_initial_indices
from src.config import DEVICE, RESULTS_DIR, PHASE2_CONFIG, CACHE_DIR
from src.simclr import SimCLR, SimCLRDatasetWrapper, train_simclr
from src.acquisition import cluster_and_select
import argparse
from torch.utils.data import DataLoader
import torch
import time
import numpy as np


def pretrain_cnn_supervised(
    train_dataset,
    labeled_indices: list,
    config: dict = {},
    device: torch.device = DEVICE,
) -> BayesianCNN:
    """
    Pretrain a CNN using supervised learning on labeled samples.
    """
    epochs = config.get("pretrain_epochs", 50)
    lr = config.get("pretrain_lr", 0.001)
    batch_size = config.get("pretrain_batch_size", 32)
    weight_decay = config.get("pretrain_weight_decay", 1e-4)

    pretrain_subset = torch.utils.data.Subset(train_dataset, list(labeled_indices))
    pretrain_loader = DataLoader(pretrain_subset, batch_size=batch_size, shuffle=True)

    cnn = BayesianCNN().to(device)
    train_model(
        cnn,
        pretrain_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    return cnn


def pretrain_cnn_simclr(
    train_dataset,
    all_indices: list,
    config: dict = {},
    device: torch.device = DEVICE,
) -> BayesianCNN:
    """
    Pretrain a CNN using SimCLR self-supervised learning.
    Caches weights to simclr_pretrained_cnn.pt.
    Need to delete file manually if hyperparams change.
    """
    epochs = config.get("simclr_epochs", 50)
    lr = config.get("simclr_lr", 0.001)
    batch_size = config.get("simclr_batch_size", 256)
    weight_decay = config.get("pretrain_weight_decay", 1e-4)
    temperature = config.get("simclr_temperature", 0.5)
    projection_dim = config.get("simclr_projection_dim", 64)
    feature_dim = config.get("feature_dim", 128)

    cache_path = CACHE_DIR / "simclr_pretrained_cnn.pt"
    cnn = BayesianCNN().to(device)

    if cache_path.exists():
        print(f"Loading cached SimCLR weights from {cache_path}")
        cnn.load_state_dict(
            torch.load(cache_path, map_location=device, weights_only=True)
        )
    else:
        subset = torch.utils.data.Subset(train_dataset, list(all_indices))
        simclr_dataset = SimCLRDatasetWrapper(subset)
        simclr_loader = DataLoader(
            simclr_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        model = SimCLR(
            backbone=cnn, feature_dim=feature_dim, projection_dim=projection_dim
        ).to(device)

        print(f"Starting SimCLR pretraining on {len(all_indices)} samples...")

        train_simclr(
            model=model,
            dataloader=simclr_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            temperature=temperature,
            verbose=True,
        )

        cnn = model.backbone
        print(f"Saving SimCLR weights to {cache_path}")
        torch.save(cnn.state_dict(), cache_path)

    return cnn


def evaluate_regression(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Evaluate multivariate regression predictions.
    Takes Nx10 tensor predictions and Nx1 tensor targets.
    Returns rmse and accuracy.
    """
    predictions = predictions.cpu()
    targets = targets.cpu()

    # argmax of predictions to get predicted class
    pred_classes = predictions.argmax(dim=1)
    accuracy = (pred_classes == targets).float().mean().item()

    # compare predictions against one-hot encoded targets
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=10).float()
    mse = ((predictions - targets_one_hot) ** 2).mean()
    rmse = torch.sqrt(mse).item()

    return rmse, accuracy


def extract_and_normalize_features(
    state: dict,
    cnn: BayesianCNN,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Extract features using a pretrained CNN and update state dict with said features.
    """
    train_features, train_labels = extract_features(cnn, train_loader, device)
    test_features, test_labels = extract_features(cnn, test_loader, device)

    # Normalisation
    feat_mean = train_features.mean(dim=0, keepdim=True)
    feat_std = train_features.std(dim=0, keepdim=True) + 1e-8
    state["train_features"] = (train_features - feat_mean) / feat_std
    state["test_features"] = (test_features - feat_mean) / feat_std
    state["train_labels"] = train_labels
    state["test_labels"] = test_labels


def fit_and_evaluate(
    state: dict,
    indices_list: list,
    method: str,
    config: dict,
    device: torch.device,
) -> tuple:
    """
    Fit regression model on labeled data, evaluate on test set.
    """
    train_features = state["train_features"]
    test_features = state["test_features"]
    train_labels = state["train_labels"]
    test_labels = state["test_labels"]

    X_train = train_features[indices_list].to(device)
    y_train_int = train_labels[indices_list].to(device)
    y_train = torch.nn.functional.one_hot(y_train_int, num_classes=10).float()
    X_test = test_features.to(device)

    start_time = time.time()

    # Full covariance matrix (dependent output dimensions)
    if method == "blr":
        model = MatrixNormalRegression(
            input_dim=config["feature_dim"],
            output_dim=10,
            prior_var=config["prior_var"],
            noise_cov=None,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        reg_time = time.time() - start_time
    # Independent output dimesnions
    elif method == "independent_blr":
        model = IndependentBLR(
            input_dim=config["feature_dim"],
            output_dim=10,
            prior_var=config["prior_var"],
            noise_var=config["noise_var"],
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test, return_std=False)
        reg_time = time.time() - start_time
    else:  # mfvi
        model = MFVIRegression(
            input_dim=config["feature_dim"],
            output_dim=10,
            prior_var=config["prior_var"],
            noise_var=config["noise_var"],
        ).to(device)
        model.fit(X_train, y_train, n_iterations=config["mfvi_epochs"])
        predictions, _ = model.predict_analytic(X_test)
        reg_time = time.time() - start_time

    rmse, accuracy = evaluate_regression(predictions, test_labels)
    return model, rmse, accuracy, reg_time


def acquire_samples(
    state: dict,
    model,
    pool_list: list,
    batch_size: int,
    method: str,
    acquisition_method: str,
    device: torch.device,
    top_n_multiplier: int = 5,
) -> list:
    """
    Select samples from the pool using the configured acquisition method (variance, random, k_max, k_centroids)
    """

    if acquisition_method == "random":
        selected_indices = np.random.choice(
            len(pool_list), size=batch_size, replace=False
        )
        return [pool_list[i] for i in selected_indices]

    # Fetches feature embeddings
    train_features = state["train_features"]
    pool_features = train_features[pool_list].to(device)

    with torch.inference_mode():
        if method == "blr" or method == "independent_blr":
            scores = model.get_acquisition_score(pool_features, method="determinant")
        else:
            scores = model.epistemic_variance(pool_features)

    if acquisition_method in ["k_centroids", "k_max"]:
        top_n = min(top_n_multiplier * batch_size, len(pool_list))
        _, top_n_indices = torch.topk(scores, k=top_n)

        candidate_features = pool_features[top_n_indices]
        candidate_scores = scores[top_n_indices]

        # Cluster and select
        selected_in_candidates = cluster_and_select(
            candidate_features,
            candidate_scores,
            k=batch_size,
            method=acquisition_method,
        )

        # Map back to pool indices
        selected_pool_indices = top_n_indices[selected_in_candidates]
        return [pool_list[i] for i in selected_pool_indices.cpu().tolist()]

    else:  # basic topk acquisition
        _, top_k_indices = torch.topk(scores, k=batch_size)
        return [pool_list[i] for i in top_k_indices.cpu().tolist()]


def run_regression_experiment(
    method: str,
    seed: int,
    config: dict,
    pretrain_method: str = "simple_pretrain",
    acquisition_method: str = "variance",
):
    set_seed(seed)

    print(
        f"\nStarting Experiment: Regression ({method.upper()}, {pretrain_method}, {acquisition_method})"
    )
    print(f"Seed: {seed} | Device: {DEVICE}")

    output_path = (
        RESULTS_DIR
        / "phase2"
        / f"{method}_{pretrain_method}_{acquisition_method}_seed{seed}.json"
    )
    print(f"Output: {output_path}\n")

    logger = RegressionLogger(method, seed)

    train_dataset, test_dataset = get_mnist_datasets()
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    initial_indices = get_balanced_initial_indices(
        train_dataset,
        n_per_class=config["initial_samples"] // 10,
        num_classes=10,
        seed=seed,
    )
    labeled_indices = set(initial_indices)
    pool_indices = set(range(len(train_dataset))) - labeled_indices

    print(
        f"Running {method.upper()} with variance-based active learning ({pretrain_method})..."
    )

    # Mutable state for features (updated when CNN is retrained)
    state: dict[str, torch.Tensor | None] = {
        "train_features": None,
        "test_features": None,
        "train_labels": None,
        "test_labels": None,
    }

    # Initial cycle
    indices_list = list(labeled_indices)
    pool_list = list(pool_indices)

    # Pretraining
    cnn_start = time.time()
    if pretrain_method == "simclr":
        print("Training CNN with SimCLR on entire training set...")
        all_indices = indices_list + pool_list
        cnn = pretrain_cnn_simclr(train_dataset, all_indices, config, DEVICE)
    else:
        print("Training CNN on initial labeled data...")
        cnn = pretrain_cnn_supervised(train_dataset, indices_list, config, DEVICE)
    cnn_time = time.time() - cnn_start

    extract_and_normalize_features(state, cnn, train_loader, test_loader, DEVICE)
    model, rmse, accuracy, reg_time = fit_and_evaluate(
        state, indices_list, method, config, DEVICE
    )

    n_samples = len(labeled_indices)
    logger.log(n_samples, rmse, accuracy)

    print(f"Cycle 0 | Samples: {n_samples}")
    print("-" * 50)
    print(f"Test RMSE     : {rmse:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.1f}%")
    print(f"Time          : CNN {format_time(cnn_time)} | Reg {format_time(reg_time)}")
    print("-" * 50 + "\n")

    # Active learning loop
    for cycle in range(1, config["acquisition_steps"] + 1):
        # Acquire samples
        pool_list = list(pool_indices)
        acq_start = time.time()
        acquired_indices = acquire_samples(
            state,
            model,
            pool_list,
            config["batch_size"],
            method,
            acquisition_method,
            DEVICE,
            top_n_multiplier=config.get("top_n_multiplier", 5),
        )
        acq_time = time.time() - acq_start

        # Log acquired indices and their targets
        acquired_targets = [train_dataset.targets[i].item() for i in acquired_indices]
        logger.log_acquisition(cycle, acquired_indices, acquired_targets)

        # Update pool and training sets
        for idx in acquired_indices:
            pool_indices.remove(idx)
            labeled_indices.add(idx)

        indices_list = list(labeled_indices)
        pool_list = list(pool_indices)

        # Under simple_retrain we have to retrain the cnn at start of each cycle
        if pretrain_method == "simple_retrain":
            cnn_start = time.time()
            cnn = pretrain_cnn_supervised(train_dataset, indices_list, config, DEVICE)
            cnn_time = time.time() - cnn_start
            extract_and_normalize_features(
                state, cnn, train_loader, test_loader, DEVICE
            )
        else:
            cnn_time = 0.0

        # Fit regression model
        model, rmse, accuracy, reg_time = fit_and_evaluate(
            state, indices_list, method, config, DEVICE
        )

        n_samples = len(labeled_indices)
        logger.log(n_samples, rmse, accuracy)

        print(f"Cycle {cycle} | Samples: {n_samples}")
        print("-" * 50)
        print(f"Test RMSE     : {rmse:.4f}")
        print(f"Test Accuracy : {accuracy * 100:.1f}%")
        print(
            f"Time          : CNN {format_time(cnn_time)} | Reg {format_time(reg_time)} | Acq {format_time(acq_time)}"
        )
        print("-" * 50 + "\n")

    logger.save(output_path)
    print(f"\nDONE: Results saved to {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Regression Baseline")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["blr", "independent_blr", "mfvi"],
        help="Inference method",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="simple_pretrain",
        choices=["simple_pretrain", "simple_retrain", "simclr"],
        help="CNN pretraining strategy",
    )
    parser.add_argument(
        "--acquisition",
        type=str,
        default="variance",
        choices=["variance", "random", "k_centroids", "k_max"],
        help="Acquisition method",
    )
    parser.add_argument(
        "--top-n-multiplier",
        type=int,
        default=5,
        help="For clustering methods (k_max, k_centroids): N = multiplier * batch_size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    config = PHASE2_CONFIG.copy()
    config["top_n_multiplier"] = args.top_n_multiplier
    if args.quick:
        config["acquisition_steps"] = 5
        config["mfvi_epochs"] = 100
        config["pretrain_epochs"] = 10
        config["simclr_epochs"] = 10
        print("Running in QUICK mode (5 cycles)")

    run_regression_experiment(
        method=args.method,
        seed=args.seed,
        config=config,
        pretrain_method=args.pretrain,
        acquisition_method=args.acquisition,
    )
