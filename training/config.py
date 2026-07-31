"""学習パイプラインの設定管理モジュール。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """学習パイプラインのハイパーパラメータとパス設定を管理する。"""

    # --- パス設定 ---
    data_dir: Path = field(default_factory=lambda: Path("data/ESC-50-master"))
    """ESC-50 データセットの展開先ディレクトリ。"""

    output_model_path: Path = field(default_factory=lambda: Path("models/bark_model.onnx"))
    """学習済みモデルの ONNX 出力先パス。"""

    # --- データセット設定 ---
    positive_classes: list[int] = field(default_factory=lambda: [0])
    """正例とする ESC-50 クラス番号のリスト。0=dog (bark/howl)。"""

    negative_classes: list[int] = field(
        default_factory=lambda: [5, 20, 21, 22, 23, 24, 26, 30]
    )
    """負例とする ESC-50 クラス番号のリスト。

    5=cat, 20=crying_baby, 21=sneezing, 22=clapping,
    23=breathing, 24=coughing, 26=laughing, 30=door_wood_knock
    """

    val_fold: int = 5
    """バリデーションに使用する ESC-50 fold 番号 (1-5)。"""

    # --- 音声前処理設定 ---
    sample_rate: int = 16000
    """ターゲットサンプリングレート (Hz)。"""

    clip_duration_sec: float = 2.0
    """学習時のランダムクロップ長（秒）。"""

    # --- 学習ハイパーパラメータ ---
    epochs: int = 50
    """学習エポック数。"""

    batch_size: int = 32
    """バッチサイズ。"""

    learning_rate: float = 1e-3
    """学習率。"""

    # --- モデル・拡張設定 ---
    fixed_frame_length: int = 199
    """固定 MFCC フレーム数。1 以上の正の整数。"""

    dropout_rate: float = 0.3
    """Dropout 率。0.0 以上 1.0 未満。"""

    use_augmentation: bool = True
    """データ拡張の有効/無効。"""

    model_type: str = "conv1d"
    """モデルタイプ。"conv1d"（可変長、CLI 推論向け）または "conv2d"（固定長、CoreML 変換向け）。"""

    augmentation_probability: float = 0.5
    """各データ拡張の適用確率。0.0 以上 1.0 以下。"""

    # --- その他 ---
    random_seed: int = 42
    """再現性のための乱数シード。"""

    @property
    def clip_length_samples(self) -> int:
        """クロップ長をサンプル数で返す。"""
        return int(self.sample_rate * self.clip_duration_sec)

    @property
    def audio_dir(self) -> Path:
        """ESC-50 音声ファイルのディレクトリパスを返す。"""
        return self.data_dir / "audio"

    @property
    def meta_csv_path(self) -> Path:
        """ESC-50 メタデータ CSV のパスを返す。"""
        return self.data_dir / "meta" / "esc50.csv"
