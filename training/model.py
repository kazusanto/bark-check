"""犬の吠え声検出用の軽量 1D CNN モデル定義。"""

from __future__ import annotations

import torch
import torch.nn as nn


class BarkCNN(nn.Module):
    """犬の吠え声を検出する軽量 1D CNN。

    入力: [batch, T, 40] (MFCC 特徴量)
    出力: [batch, 1] (吠え声確率, 0.0〜1.0)

    構成:
        Conv1d(40, 64) → BN → ReLU → MaxPool1d →
        Conv1d(64, 128) → BN → ReLU → MaxPool1d →
        Conv1d(128, 128) → BN → ReLU → Global Average Pooling →
        Dropout → Linear(128, 1) → Sigmoid
    """

    def __init__(self, dropout_rate: float = 0.3) -> None:
        """BarkCNN を初期化する。

        Args:
            dropout_rate: Dropout 率。0.0 以上 1.0 未満。
        """
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

        self.dropout = nn.Dropout(dropout_rate)

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

        # Dropout
        x = self.dropout(x)

        # 分類器
        x = self.classifier(x)

        return x


class BarkCNN2d(nn.Module):
    """CoreML 互換の Conv2d ベース犬の吠え声検出モデル。

    入力: [batch, 1, 40, 199] (4D, MFCC 特徴量)
    出力: [batch, 1] (吠え声確率, 0.0〜1.0)

    内部で入力を [B, 40, 1, 199] に permute し、Conv2d((1,3)) で
    時間軸方向に畳み込む。これは Conv1d(40, 64, 3) と数学的に等価。

    構成:
        permute [B,1,40,199] → [B,40,1,199]
        Conv2d(40, 64, (1,3), padding=(0,1)) → BN2d → ReLU → MaxPool2d((1,2))
        Conv2d(64, 128, (1,3), padding=(0,1)) → BN2d → ReLU → MaxPool2d((1,2))
        Conv2d(128, 128, (1,3), padding=(0,1)) → BN2d → ReLU
        → AdaptiveAvgPool2d((1,1))
        → Dropout(dropout_rate)
        → Linear(128, 1) → Sigmoid
    """

    def __init__(self, dropout_rate: float = 0.3) -> None:
        """BarkCNN2d を初期化する。

        Args:
            dropout_rate: Dropout 率。0.0 以上 1.0 未満。
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(40, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            # Block 3
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を実行する。

        Args:
            x: 入力テンソル, shape [batch, 1, 40, 199]。

        Returns:
            吠え声確率, shape [batch, 1]。
        """
        # [B, 1, 40, 199] → [B, 40, 1, 199]
        x = x.permute(0, 2, 1, 3)

        # Conv2d ブロック
        x = self.features(x)

        # Global Average Pooling: [B, 128, 1, T'] → [B, 128, 1, 1]
        x = self.gap(x)

        # Flatten: [B, 128, 1, 1] → [B, 128]
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

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
