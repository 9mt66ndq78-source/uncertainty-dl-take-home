import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLR(nn.Module):
    """
    Wraps a backbone network with a projection head that is used during
    pretraining and discarded afterward.
    """

    def __init__(self, backbone, feature_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        # Extract features without dropout for stable contrastive learning
        h = self.backbone.extract_features(x, apply_dropout=False)
        z = self.projector(h)
        return h, z  # Returns both feature embeddings and projections


class SimCLRDatasetWrapper(Dataset):
    """
    Wraps a dataset to return two augmented views of the same image to support contrastive learning.
    Handles un-normalization of pre-normalized MNIST images before
    applying augmentations, then re-normalizes.
    """

    # MNIST normalization constants
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081

    def __init__(self, dataset):
        self.dataset = dataset

        # Strong augmentations for MNIST
        # Applied after un-normalization, followed by re-normalization
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5
                ),
                transforms.Normalize((self.MNIST_MEAN,), (self.MNIST_STD,)),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns two different augmented views of the same image.
        """
        img, _ = self.dataset[idx]

        # Un-normalize
        img = img * self.MNIST_STD + self.MNIST_MEAN

        # Clamp to valid range
        img = torch.clamp(img, 0.0, 1.0)

        # Apply random augmentations twice for two views
        x_i = self.transform(img)
        x_j = self.transform(img)

        return x_i, x_j


def nt_xent_loss(
    z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """
    Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    """
    batch_size = z_i.shape[0]
    device = z_i.device

    # L2 normalize projections
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate
    z = torch.cat([z_i, z_j], dim=0)

    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    targets = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ]
    )

    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, targets)

    return loss


def train_simclr(
    model: SimCLR,
    dataloader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    Train a SimCLR model with NT-Xent loss.
    """
    from tqdm import tqdm

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    epoch_iter = tqdm(
        range(epochs), desc="SimCLR Epochs", disable=not verbose, position=0
    )

    for epoch in epoch_iter:
        total_loss = 0.0
        num_batches = 0

        batch_iter = tqdm(
            dataloader,
            desc=f"  Epoch {epoch + 1:>3}",
            disable=not verbose,
            position=1,
            leave=False,
        )

        for x_i, x_j in batch_iter:
            x_i, x_j = x_i.to(device), x_j.to(device)

            optimizer.zero_grad()

            # Forward pass for both views
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Compute contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update batch progress bar with current loss
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_iter.set_postfix(avg_loss=f"{avg_loss:.4f}")
