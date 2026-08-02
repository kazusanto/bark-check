"""UrbanSound8K データソースに特化した Dataset モジュール。"""

from __future__ import annotations

import csv

from training.config import TrainingConfig
from training.dataset_base import BarkDatasetBase


class UrbanSound8KBarkDataset(BarkDatasetBase):
    """UrbanSound8K データソースに特化した Dataset。"""

    def _get_val_fold(self) -> int:
        """バリデーション fold 番号を返す。"""
        return self._config.urbansound8k_val_fold

    def load_entries(self) -> list[dict]:
        """UrbanSound8K メタデータ CSV を読み込み、対象クラスのエントリリストを返す。

        Returns:
            各エントリは {"filepath": Path, "label": int, "fold": int} を含む。
            label は 1（正例）または 0（負例）。
        """
        all_classes = set(
            self._config.urbansound8k_positive_classes
            + self._config.urbansound8k_negative_classes
        )
        entries = []
        csv_path = self._config.urbansound8k_dir / "metadata" / "UrbanSound8K.csv"

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row["classID"])
                if class_id not in all_classes:
                    continue
                label = (
                    1
                    if class_id in self._config.urbansound8k_positive_classes
                    else 0
                )
                fold = int(row["fold"])
                filepath = (
                    self._config.urbansound8k_dir
                    / "audio"
                    / f"fold{fold}"
                    / row["slice_file_name"]
                )
                entries.append({
                    "filepath": filepath,
                    "label": label,
                    "fold": fold,
                })
        return entries
