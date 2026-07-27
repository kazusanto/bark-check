"""犬の吠え声検出用の軽量 1D CNN モデル定義。"""

from __future__ import annotations

import torch
import torch.nn as nn


class BarkCNN(nn.Module):
    """犬の吠え声を検出する軽量 1D CNN。

    入力: [batch, T, 40] (MFCC 特徴量)
    出力: [batch, 1] (吠え声確率, 0.0〜1.0)

    構成:
        Conv1d(40, 64) → BN → ReLU → Conv1d(64, 128) → BN → ReLU →
        Conv1d(128, 128) → BN → ReLU → Global Average Pooling →
        Linear(128, 1) → Sigmoid
    """

    def __init__(self) -> None:
        """BarkCNN を初期化する。"""
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を実行する。

        Args:
            x: 入力テンソル, shape [batch, T, 40]。

        Returns:
            吠え声確率, shape [batch, 1]。
        """
        # [batch, T, 40] → [batch, 40, T] (Conv1d は channels first)
        x = x.permute(0, 2, 1)

        # 畳み込みブロック
        x = self.features(x)

        # Global Average Pooling: [batch, 128, T'] → [batch, 128]
        x = x.mean(dim=2)

        # 分類器
        x = self.classifier(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """モデルの学習可能パラメータ数を返す。

    Args:
        model: PyTorch モデル。

    Returns:
        学習可能パラメータの総数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
