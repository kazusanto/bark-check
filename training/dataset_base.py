"""複数データソースに対応する抽象基底データセットモジュール。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bark_check.feature_extractor import FeatureExtractor
from training.augmentation import apply_gaussian_noise, apply_time_shift
from training.config import TrainingConfig


class BarkDatasetBase(Dataset, ABC):
    """複数データソースに対応する抽象基底データセット。

    音声読み込み→クロップ/パディング→拡張→MFCC 抽出の共通パイプラインを集約する。
    サブクラスは load_entries() と _get_val_fold() を実装する。
    """

    def __init__(self, config: TrainingConfig, *, is_train: bool = True) -> None:
        """BarkDatasetBase を初期化する。

        Args:
            config: 学習設定。
            is_train: True の場合 val_fold 以外を使用。False の場合 val_fold のみ使用。
        """
        self._config = config
        self._is_train = is_train
        self._feature_extractor = FeatureExtractor()
        self._entries: list[dict] = []

        all_entries = self.load_entries()

        # fold-based split
        val_fold = self._get_val_fold()
        if is_train:
            self._entries = [e for e in all_entries if e["fold"] != val_fold]
        else:
            self._entries = [e for e in all_entries if e["fold"] == val_fold]

    @abstractmethod
    def load_entries(self) -> list[dict]:
        """データソース固有のメタデータを読み込み、統一形式のエントリリストを返す。

        Returns:
            各エントリは {"filepath": Path, "label": int, "fold": int} を含む。
            label は 1（正例）または 0（負例）。
            fold は 1 以上の正の整数。
        """
        ...

    @abstractmethod
    def _get_val_fold(self) -> int:
        """バリデーション fold 番号を返す。"""
        ...

    def __len__(self) -> int:
        """データセットのサンプル数を返す。"""
        return len(self._entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """指定インデックスのサンプルを返す。

        Args:
            idx: データセット内のインデックス。

        Returns:
            (features, label) のタプル。
            model_type == "conv2d" の場合:
                features: shape [1, 40, fixed_frame_length] の MFCC テンソル (float32)。
            model_type == "conv1d" の場合:
                features: shape [T, 40] の MFCC テンソル (float32)。
            label: shape [1] のラベルテンソル (float32, 0.0 or 1.0)。

        Raises:
            FileNotFoundError: 音声ファイルが存在しない場合。
        """
        entry = self._entries[idx]
        filepath: Path = entry["filepath"]

        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        import librosa

        audio, sr = librosa.load(str(filepath), sr=self._config.sample_rate, mono=True)

        # crop/pad
        clip_samples = self._config.clip_length_samples
        if len(audio) > clip_samples:
            if self._is_train:
                start = np.random.randint(0, len(audio) - clip_samples)
            else:
                start = (len(audio) - clip_samples) // 2
            audio = audio[start : start + clip_samples]
        else:
            pad_length = clip_samples - len(audio)
            audio = np.pad(audio, (0, pad_length), mode="constant")

        pcm = audio.astype(np.float32)

        if self._config.model_type == "conv2d":
            # データ拡張（学習時のみ）
            if self._is_train and self._config.use_augmentation:
                if np.random.random() < self._config.augmentation_probability:
                    pcm = apply_time_shift(pcm)
                if np.random.random() < self._config.augmentation_probability:
                    pcm = apply_gaussian_noise(pcm)

            # 固定長 MFCC 特徴量抽出
            features = self._feature_extractor.extract(
                pcm,
                self._config.sample_rate,
                fixed_length=self._config.fixed_frame_length,
            )

            # channels-first に転置: [fixed_frame_length, 40] → [40, fixed_frame_length]
            features_tensor = torch.from_numpy(features).T
            # チャンネル次元を追加: [40, fixed_frame_length] → [1, 40, fixed_frame_length]
            features_tensor = features_tensor.unsqueeze(0)
        else:
            # conv1d パス: 可変長 MFCC（データ拡張なし、fixed_length なし）
            features = self._feature_extractor.extract(pcm, self._config.sample_rate)
            features_tensor = torch.from_numpy(features)  # [T, 40]

        label_tensor = torch.tensor([entry["label"]], dtype=torch.float32)  # [1]

        return features_tensor, label_tensor
