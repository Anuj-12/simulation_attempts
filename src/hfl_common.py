"""
Shared model, dataset, and configuration utilities for the ReCon HFL simulation.

This module is the leaf of the dependency graph — it must not import from
hfl_base or hfl_rl.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor


class FashionMNISTNet(nn.Module):
    """Small CNN shared by all edge UAVs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


FASHION_MNIST_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.2860,), (0.3530,)),
])


def load_fashion_mnist(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    train = FashionMNIST(root=data_dir, train=True, download=True, transform=FASHION_MNIST_TRANSFORM)
    test = FashionMNIST(root=data_dir, train=False, download=True, transform=FASHION_MNIST_TRANSFORM)
    return train, test


def partition_dataset(dataset: Dataset, num_clients: int) -> List[Subset]:
    """Split a dataset into num_clients roughly equal non-overlapping shards."""
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    shard_size = len(dataset) // num_clients
    shards: List[Subset] = []
    for i in range(num_clients):
        start = i * shard_size
        end = len(dataset) if i == num_clients - 1 else (i + 1) * shard_size
        shards.append(Subset(dataset, indices[start:end]))
    return shards


@dataclass
class HFLConfig:
    data_dir: str = "./data"
    num_rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    device: str = "cpu"
