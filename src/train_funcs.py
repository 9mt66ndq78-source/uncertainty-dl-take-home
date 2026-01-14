import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Tuple


def train_model(
    model,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    """
    Full training loop
    """
    start_time = time.time()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pbar = tqdm(range(1, epochs + 1), leave=False)
    avg_loss = 0.0

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x, apply_dropout=True)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        pbar.set_description(f"Epoch {epoch}/{epochs} [Loss: {avg_loss:.3f}]")

    training_time = time.time() - start_time
    return training_time


def evaluate_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    mc_samples: int = 100,
    deterministic: bool = False,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.NLLLoss(reduction="sum")

    correct = 0
    total = 0
    total_nll = 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if deterministic:
                # Single forward pass with dropout disabled
                log_probs = model(batch_x, apply_dropout=False)
                probs = torch.exp(log_probs)
            else:
                # Get predictions using MC Dropout
                mean_probs, _ = model.predict_proba(
                    batch_x, n_samples=mc_samples, return_samples=False
                )
                probs = mean_probs
                log_probs = torch.log(mean_probs + 1e-10)

            # Accuracy
            predictions = probs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            # NLL
            nll = criterion(log_probs, batch_y)
            total_nll += nll.item()

    test_accuracy = correct / total
    test_nll = total_nll / total

    return test_accuracy, test_nll


def reset_model(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def extract_features(model, dataloader: DataLoader, device: torch.device):
    """
    Extract features from the penultimate layer of a trained CNN.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            features = model.extract_features(images, apply_dropout=False)
            all_features.append(features.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
