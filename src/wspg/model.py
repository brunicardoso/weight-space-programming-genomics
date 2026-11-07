"""SmallCNN: the target model. Input: (B, 4, L) one-hot in A,C,G,T order. Output: (B,) logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, n_filters: int = 64, w1: int = 19, w2: int = 7, w3: int = 5, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, w1, padding="same")
        self.conv2 = nn.Conv1d(n_filters, n_filters, w2, padding="same")
        self.conv3 = nn.Conv1d(n_filters, n_filters, w3, padding="same")
        self.fc1 = nn.Linear(n_filters, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool1d(F.relu(self.conv1(x)), 4)
        x = F.max_pool1d(F.relu(self.conv2(x)), 4)
        x = F.relu(self.conv3(x)).amax(dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def build_model(cfg: dict | None = None) -> SmallCNN:
    return SmallCNN(**(cfg or {}))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
