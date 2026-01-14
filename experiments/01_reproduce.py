import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse

from src.config import DEVICE, PHASE1_CONFIG, RESULTS_DIR
from src.data_funcs import (
    get_mnist_datasets,
    get_balanced_initial_indices,
    get_validation_indices,
)
from src.models import BayesianCNN
from src.train_funcs import train_model, evaluate_model
from src.acquisition import acquire_batch
from src.utils import set_seed, JSONLogger, format_time


# Tunes weight decay on validation set
def tune_weight_decay(
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight_decay_grid: list,
    config: dict,
    device: torch.device,
) -> float:
    best_wd = weight_decay_grid[0]
    best_nll = float("inf")

    for wd in weight_decay_grid:
        model = BayesianCNN().to(device)
        train_model(
            model,
            train_loader,
            device,
            epochs=config["epochs"],
            lr=config["learning_rate"],
            weight_decay=wd,
        )
        _, val_nll = evaluate_model(
            model, val_loader, device, mc_samples=config["mc_samples"]
        )

        if val_nll < best_nll:
            best_nll = val_nll
            best_wd = wd

    return best_wd


def run_active_learning(
    method: str,
    seed: int,
    config: dict,
    deterministic: bool = False,
):
    set_seed(seed)
    rng = np.random.RandomState(seed)

    print(
        f"\nStarting Experiment (Method: {method.upper()}, Deterministic: {deterministic})"
    )
    print(f"Seed: {seed}, Device: {DEVICE}")
    if deterministic:
        output_path = RESULTS_DIR / "phase1" / f"{method}_deterministic_seed{seed}.json"
    else:
        output_path = RESULTS_DIR / "phase1" / f"{method}_seed{seed}.json"
    print(f"Output: {output_path}\n")

    json_logger = JSONLogger(method, seed)

    train_dataset, test_dataset = get_mnist_datasets()
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    initial_indices = get_balanced_initial_indices(
        train_dataset,
        n_per_class=config["samples_per_class"],
        num_classes=10,
        seed=seed,
    )
    val_indices = get_validation_indices(
        train_dataset,
        n_total=config["val_size"],
        exclude_indices=initial_indices,
        seed=seed,
    )

    labeled_indices = set(initial_indices)
    val_indices_set = set(val_indices)
    pool_indices = set(range(len(train_dataset))) - labeled_indices - val_indices_set

    val_loader = DataLoader(
        Subset(train_dataset, val_indices), batch_size=128, shuffle=False
    )

    model = BayesianCNN().to(DEVICE)

    # Initial weight decay tuning
    print("Tuning weight decay on validation set...")
    labeled_loader = DataLoader(
        Subset(train_dataset, list(labeled_indices)), batch_size=32, shuffle=True
    )
    weight_decay = tune_weight_decay(
        labeled_loader, val_loader, config["weight_decay_grid"], config, DEVICE
    )
    print(f"Selected weight decay: {weight_decay}\n")

    model = BayesianCNN().to(DEVICE)
    train_time = train_model(
        model,
        labeled_loader,
        DEVICE,
        epochs=config["epochs"],
        lr=config["learning_rate"],
        weight_decay=weight_decay,
    )
    test_acc, test_nll = evaluate_model(
        model, test_loader, DEVICE, config["mc_samples"], deterministic=deterministic
    )

    # Log initial results
    n_samples = len(labeled_indices)
    json_logger.log_cycle(n_samples, test_acc, test_nll)

    print(f"Cycle 0 | Samples: {n_samples}")
    print("-" * 50)
    print(f"Test Accuracy : {test_acc * 100:.1f}%")
    print(f"Test NLL      : {test_nll:.3f}")
    print(f"Time          : Train {format_time(train_time)}")
    print("-" * 50 + "\n")

    # Active learning loop
    for cycle in range(1, config["acquisition_steps"] + 1):
        pool_list = list(pool_indices)
        pool_loader = DataLoader(
            Subset(train_dataset, pool_list), batch_size=1024, shuffle=False
        )

        selected_indices, acq_time = acquire_batch(
            model,
            pool_loader,
            batch_size=config["batch_size"],
            method=method,
            mc_samples=config["mc_samples_acq"],
            device=DEVICE,
            rng=rng,
            deterministic=deterministic,
        )

        # Map pool indices back to dataset indices
        dataset_indices = [pool_list[i] for i in selected_indices]

        # Log acquired indices and their targets
        acquired_targets = [train_dataset.targets[i].item() for i in dataset_indices]
        json_logger.log_acquisition(cycle, dataset_indices, acquired_targets)

        for idx in dataset_indices:
            pool_indices.remove(idx)
            labeled_indices.add(idx)

        print(f"\nCycle {cycle}/{config['acquisition_steps']}")
        model = BayesianCNN().to(DEVICE)

        labeled_loader = DataLoader(
            Subset(train_dataset, list(labeled_indices)), batch_size=32, shuffle=True
        )

        # Periodically retune weight decay
        retune_every = config.get("retune_every", 0)
        if retune_every > 0 and cycle % retune_every == 0:
            print("Retuning weight decay...")
            weight_decay = tune_weight_decay(
                labeled_loader, val_loader, config["weight_decay_grid"], config, DEVICE
            )
            print(f"New weight decay: {weight_decay}")
            model = BayesianCNN().to(DEVICE)  # Fresh model after tuning

        train_time = train_model(
            model,
            labeled_loader,
            DEVICE,
            epochs=config["epochs"],
            lr=config["learning_rate"],
            weight_decay=weight_decay,
        )

        # Evaluation step
        eval_start = time.time()
        test_acc, test_nll = evaluate_model(
            model,
            test_loader,
            DEVICE,
            config["mc_samples"],
            deterministic=deterministic,
        )
        eval_time = time.time() - eval_start

        n_samples = len(labeled_indices)
        json_logger.log_cycle(n_samples, test_acc, test_nll)

        print(f"\nCycle {cycle} | Samples: {n_samples}")
        print("-" * 50)
        print(f"Test Accuracy : {test_acc * 100:.1f}%")
        print(f"Test NLL      : {test_nll:.3f}")
        print(
            f"Time          : Train {format_time(train_time)}, Acq {format_time(acq_time)}, Eval {format_time(eval_time)}"
        )
        print("-" * 50)

    json_logger.save(output_path)

    print(f"\nDONE: Results saved to {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Classification Reproduction")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["random", "bald", "entropy", "variation_ratios", "mean_std"],
        help="Acquisition method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic evaluation (no MC sampling at acquisition and test time)",
    )

    args = parser.parse_args()

    config = PHASE1_CONFIG.copy()
    if args.quick:
        config["acquisition_steps"] = 5
        config["epochs"] = 5
        print("Running in QUICK mode (5 cycles, 5 epochs)")

    run_active_learning(
        method=args.method,
        seed=args.seed,
        config=config,
        deterministic=args.deterministic,
    )
