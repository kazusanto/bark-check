"""TrainingConfig に基づいてデータセットを構築するファクトリモジュール。"""

from __future__ import annotations

from torch.utils.data import ConcatDataset, Dataset

from training.config import TrainingConfig
from training.dataset import ESC50BarkDataset
from training.dataset_base import BarkDatasetBase
from training.dataset_urbansound8k import UrbanSound8KBarkDataset

_VALID_SOURCES = {"esc50", "urbansound8k"}


def build_dataset(config: TrainingConfig, *, is_train: bool) -> Dataset:
    """TrainingConfig に基づいてデータセットを構築する。

    Args:
        config: 学習設定。
        is_train: True で学習用、False でバリデーション用。

    Returns:
        構築された Dataset（単一 or ConcatDataset）。

    Raises:
        ValueError: 無効なデータソース名、クラス重複、クラスID 範囲外の場合。
        FileNotFoundError: UrbanSound8K メタデータが見つからない場合。
    """
    _validate_config(config)

    datasets: list[Dataset] = []
    for source in config.data_sources:
        if source == "esc50":
            datasets.append(ESC50BarkDataset(config, is_train=is_train))
        elif source == "urbansound8k":
            _check_urbansound8k_available(config)
            datasets.append(UrbanSound8KBarkDataset(config, is_train=is_train))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    if is_train:
        dataset = _apply_class_balance(dataset)

    return dataset


def _validate_config(config: TrainingConfig) -> None:
    """data_sources とクラス設定のバリデーションを行う。

    Args:
        config: 検証対象の学習設定。

    Raises:
        ValueError: 未知のデータソース名、classID 範囲外、positive/negative 重複の場合。
    """
    for source in config.data_sources:
        if source not in _VALID_SOURCES:
            raise ValueError(
                f"Unknown data source: '{source}'. "
                f"Valid sources: {sorted(_VALID_SOURCES)}"
            )

    if "urbansound8k" in config.data_sources:
        all_us8k_classes = (
            config.urbansound8k_positive_classes
            + config.urbansound8k_negative_classes
        )
        for cid in all_us8k_classes:
            if not (0 <= cid <= 9):
                raise ValueError(
                    f"Invalid UrbanSound8K classID: {cid}. Must be 0-9."
                )

        overlap = set(config.urbansound8k_positive_classes) & set(
            config.urbansound8k_negative_classes
        )
        if overlap:
            raise ValueError(
                f"UrbanSound8K positive/negative class overlap: {sorted(overlap)}"
            )


def _check_urbansound8k_available(config: TrainingConfig) -> None:
    """UrbanSound8K メタデータ CSV の存在を確認する。

    Args:
        config: 学習設定。

    Raises:
        FileNotFoundError: メタデータ CSV が見つからない場合。
            エラーメッセージにダウンロード URL と期待ディレクトリ構成を含める。
    """
    csv_path = config.urbansound8k_dir / "metadata" / "UrbanSound8K.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"UrbanSound8K metadata not found: {csv_path}\n"
            f"Download from: https://urbansounddataset.weebly.com/urbansound8k.html\n"
            f"Expected structure:\n"
            f"  {config.urbansound8k_dir}/metadata/UrbanSound8K.csv\n"
            f"  {config.urbansound8k_dir}/audio/fold1/ ~ fold10/"
        )


def _apply_class_balance(dataset: Dataset) -> Dataset:
    """正例のオーバーサンプリングでクラスバランスを調整する。

    repeat_factor + remainder 方式で正例数を負例数と一致させる。
    is_train=True のときのみ呼び出される前提。

    Args:
        dataset: クラスバランス調整対象のデータセット。

    Returns:
        バランス調整後のデータセット。正例または負例のみの場合は調整せず返す。
    """
    if isinstance(dataset, BarkDatasetBase):
        entries = dataset._entries
    elif isinstance(dataset, ConcatDataset):
        entries = []
        for ds in dataset.datasets:
            entries.extend(ds._entries)  # type: ignore[attr-defined]
    else:
        return dataset

    positives = [e for e in entries if e["label"] == 1]
    negatives = [e for e in entries if e["label"] == 0]

    if not positives or not negatives:
        return dataset

    repeat_factor = len(negatives) // len(positives)
    remainder = len(negatives) % len(positives)
    balanced_positives = positives * repeat_factor + positives[:remainder]
    balanced_entries = balanced_positives + negatives

    if isinstance(dataset, BarkDatasetBase):
        dataset._entries = balanced_entries
        return dataset

    # ConcatDataset の場合、最初のサブデータセットにバランス後エントリを集約する
    first_ds = dataset.datasets[0]
    first_ds._entries = balanced_entries  # type: ignore[attr-defined]
    return first_ds
