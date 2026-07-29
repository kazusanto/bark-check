"""ESC-50 データセットのダウンロード・前処理・PyTorch Dataset を提供する。"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bark_check.feature_extractor import FeatureExtractor
from training.augmentation import apply_gaussian_noise, apply_time_shift
from training.config import TrainingConfig

# ESC-50 リポジトリの ZIP ダウンロード URL
_ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def download_esc50(data_dir: Path) -> None:
    """ESC-50 データセットをダウンロードして展開する。

    既に展開済みの場合はスキップする。

    Args:
        data_dir: ESC-50 の展開先ディレクトリ（例: data/ESC-50-master）。
    """
    if data_dir.exists() and (data_dir / "meta" / "esc50.csv").exists():
        print(f"ESC-50 は既にダウンロード済み: {data_dir}")
        return

    print(f"ESC-50 をダウンロード中... ({_ESC50_URL})")
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    response = urllib.request.urlopen(_ESC50_URL)  # noqa: S310
    zip_bytes = io.BytesIO(response.read())

    print("展開中...")
    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extractall(data_dir.parent)

    print(f"ESC-50 ダウンロード完了: {data_dir}")


def _load_metadata(config: TrainingConfig) -> list[dict]:
    """ESC-50 メタデータ CSV を読み込み、対象クラスのみフィルタして返す。

    Args:
        config: 学習設定。

    Returns:
        対象クラスのエントリリスト。各エントリは
        {"filename": str, "fold": int, "target": int, "label": int} を含む。
        label は 1（正例）または 0（負例）。
    """
    all_classes = set(config.positive_classes + config.negative_classes)
    entries = []

    with open(config.meta_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = int(row["target"])
            if target not in all_classes:
                continue

            label = 1 if target in config.positive_classes else 0
            entries.append(
                {
                    "filename": row["filename"],
                    "fold": int(row["fold"]),
                    "target": target,
                    "label": label,
                }
            )

    return entries


class ESC50BarkDataset(Dataset):
    """ESC-50 から構築した犬の吠え声二値分類データセット。

    各サンプルは 5 秒のクリップからランダムクロップ（デフォルト 2 秒）して
    MFCC 特徴量を抽出したテンソルとラベルのペアを返す。
    """

    def __init__(
        self,
        config: TrainingConfig,
        *,
        is_train: bool = True,
    ) -> None:
        """ESC50BarkDataset を初期化する。

        Args:
            config: 学習設定。
            is_train: True の場合 val_fold 以外を使用。False の場合 val_fold のみ使用。
        """
        self._config = config
        self._is_train = is_train
        self._feature_extractor = FeatureExtractor()

        all_entries = _load_metadata(config)

        if is_train:
            self._entries = [e for e in all_entries if e["fold"] != config.val_fold]
        else:
            self._entries = [e for e in all_entries if e["fold"] == config.val_fold]

        # 学習時は正例のオーバーサンプリングでクラスバランスを取る
        if is_train:
            self._entries = self._balance_classes(self._entries)

    def _balance_classes(self, entries: list[dict]) -> list[dict]:
        """正例をオーバーサンプリングしてクラスバランスを取る。

        Args:
            entries: 元のエントリリスト。

        Returns:
            バランス調整後のエントリリスト。
        """
        positives = [e for e in entries if e["label"] == 1]
        negatives = [e for e in entries if e["label"] == 0]

        if not positives or not negatives:
            return entries

        # 負例に合わせて正例をリピート
        repeat_factor = len(negatives) // len(positives)
        remainder = len(negatives) % len(positives)

        balanced_positives = positives * repeat_factor + positives[:remainder]
        return balanced_positives + negatives

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
                features: shape [1, 40, 199] の MFCC テンソル (float32, channels-first)。
            model_type == "conv1d" の場合:
                features: shape [T, 40] の MFCC テンソル (float32)。
            label: shape [1] のラベルテンソル (float32, 0.0 or 1.0)。
        """
        entry = self._entries[idx]
        wav_path = self._config.audio_dir / entry["filename"]

        # 音声読み込み（librosa で 16kHz モノラルに変換）
        import librosa

        audio, sr = librosa.load(str(wav_path), sr=self._config.sample_rate, mono=True)

        # ランダムクロップ
        clip_samples = self._config.clip_length_samples
        if len(audio) > clip_samples:
            if self._is_train:
                start = np.random.randint(0, len(audio) - clip_samples)
            else:
                # バリデーション時は中央クロップ
                start = (len(audio) - clip_samples) // 2
            audio = audio[start : start + clip_samples]
        else:
            # 短い場合はゼロパディング
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
                pcm, self._config.sample_rate,
                fixed_length=self._config.fixed_frame_length,
            )

            # channels-first に転置: [199, 40] → [40, 199]
            features_tensor = torch.from_numpy(features).T
            # チャンネル次元を追加: [40, 199] → [1, 40, 199]
            features_tensor = features_tensor.unsqueeze(0)
        else:
            # conv1d パス: 可変長 MFCC（データ拡張なし、fixed_length なし）
            features = self._feature_extractor.extract(pcm, self._config.sample_rate)
            features_tensor = torch.from_numpy(features)  # [T, 40]

        label_tensor = torch.tensor([entry["label"]], dtype=torch.float32)  # [1]

        return features_tensor, label_tensor
