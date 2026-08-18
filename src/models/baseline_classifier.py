"""Phase 3 baseline 1D-CNN classifier for PTB-XL ECG superclass classification.

Simplified architecture for faster CPU training:
  - Stem: Conv1d(12 -> 32, kernel=7, stride=2) + BN + ReLU
  - 3 Conv blocks with pooling (no residuals)
    Block 1: 32 -> 64 channels, kernel=5, pool=2
    Block 2: 64 -> 128 channels, kernel=3, pool=2
    Block 3: 128 -> 128 channels, kernel=3, pool=2
  - Global Average Pooling over time
  - Dropout (p=0.2)
  - Linear head to 5 classes (raw logits, for BCEWithLogitsLoss)

Total parameters: ~150K (well within the 100K-5M test range).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic conv-bn-relu-pool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x)


class ECGClassifier1D(nn.Module):
    """1D-CNN baseline for multi-label ECG superclass classification.

    Returns raw logits (no sigmoid) — use with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.n_classes = n_classes

        # Stem: reduce temporal resolution early for efficiency
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Conv blocks with progressive channel expansion and pooling
        self.block1 = ConvBlock(32, 64, kernel_size=5, pool_size=2)
        self.block2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        self.block3 = ConvBlock(128, 128, kernel_size=3, pool_size=2)

        # Global average pooling + classifier head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_leads, T) where T = 5000 samples (10s @ 500Hz)
        Returns:
            logits: Tensor of shape (batch, n_classes) — raw logits, NO sigmoid
        """
        x = self.stem(x)       # (B, 32, T/2)
        x = self.block1(x)     # (B, 64, T/4)
        x = self.block2(x)     # (B, 128, T/8)
        x = self.block3(x)     # (B, 128, T/16)
        x = self.gap(x).squeeze(-1)  # (B, 128)
        x = self.dropout(x)
        logits = self.head(x)  # (B, n_classes)
        return logits