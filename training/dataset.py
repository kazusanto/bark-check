"""ESC-50 データセットのダウンロード・前処理・PyTorch Dataset を提供する。"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

from training.config import TrainingConfig
from training.dataset_base import BarkDatasetBase

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


class ESC50BarkDataset(BarkDatasetBase):
    """ESC-50 データソースに特化した Dataset。"""

    def _get_val_fold(self) -> int:
        """バリデーション fold 番号を返す。"""
        return self._config.val_fold

    def load_entries(self) -> list[dict]:
        """ESC-50 メタデータ CSV を読み込み、対象クラスのエントリリストを返す。

        Returns:
            各エントリは {"filepath": Path, "label": int, "fold": int} を含む。
            label は 1（正例）または 0（負例）。
        """
        all_classes = set(self._config.positive_classes + self._config.negative_classes)
        entries = []

        with open(self._config.meta_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = int(row["target"])
                if target not in all_classes:
                    continue
                label = 1 if target in self._config.positive_classes else 0
                entries.append({
                    "filepath": self._config.audio_dir / row["filename"],
                    "label": label,
                    "fold": int(row["fold"]),
                })
        return entries
