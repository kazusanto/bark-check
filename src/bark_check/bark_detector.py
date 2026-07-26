"""犬の吠え声を検出するコアロジックモジュール。"""

from __future__ import annotations

import time

import numpy as np

from bark_check.feature_extractor import FeatureExtractor
from bark_check.models import DetectionResult

# 入力 PCM の最大長（サンプル数）
_MAX_PCM_LENGTH = 32000


class ModelLoadError(Exception):
    """モデルの読み込みに失敗した場合に送出される例外。"""


class BarkDetector:
    """犬の吠え声を検出するコアライブラリ。CLI 依存を一切持たない。"""

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        """BarkDetector を初期化する。

        Args:
            threshold: 吠え声判定の閾値。0.0 以上 1.0 以下の範囲で指定する。
            model_path: 事前学習済み ONNX モデルのパス。None の場合はモデル未ロード状態で
                初期化される（detect() 呼び出し時に error フィールドにエラーが格納される）。

        Raises:
            ValueError: threshold が [0.0, 1.0] の範囲外の場合。
            ModelLoadError: model_path が指定されているがファイルが存在しない、
                またはモデルの読み込みに失敗した場合。
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"threshold must be between 0.0 and 1.0, got {threshold}"
            )

        self._threshold = threshold
        self._session = None
        self._model_path: str | None = model_path
        self._feature_extractor = FeatureExtractor()

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """ONNX モデルを読み込む。

        Args:
            model_path: ONNX モデルファイルのパス。

        Raises:
            ModelLoadError: ファイルが存在しない、またはモデルの読み込みに失敗した場合。
        """
        import os

        if not os.path.exists(model_path):
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def detect(self, pcm: np.ndarray, sample_rate: int) -> DetectionResult:
        """モノラル PCM データから犬の吠え声を検出する。

        例外は発生しない。エラーは DetectionResult.error に格納される。

        Args:
            pcm: float32 型のモノラル PCM サンプル配列 (shape: [N])。
            sample_rate: サンプリングレート（Hz）。

        Returns:
            is_bark, confidence, timestamp, audio_duration, error を含む判定結果。
        """
        timestamp = time.time()
        audio_duration = len(pcm) / sample_rate if sample_rate > 0 else 0.0

        # 空入力チェック
        if len(pcm) == 0:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=0.0,
                error="Input PCM block is empty",
            )

        # 上限超過チェック
        if len(pcm) > _MAX_PCM_LENGTH:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error="Input exceeds maximum length of 32000 samples",
            )

        # 無音入力チェック
        if np.all(pcm == 0):
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error=None,
            )

        # モデル未ロードチェック
        if self._session is None:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error="No model loaded",
            )

        # 推論
        try:
            features = self._feature_extractor.extract(pcm, sample_rate)
            # 入力テンソル形状: [1, T, 40]（batch=1, frames, mfcc_dims）
            input_tensor = features[np.newaxis, :, :].astype(np.float32)

            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})

            # 出力テンソル形状: [1, 1]（bark_probability）
            raw_confidence = float(outputs[0][0][0])
            confidence = float(np.clip(raw_confidence, 0.0, 1.0))
            is_bark = confidence >= self._threshold

            return DetectionResult(
                is_bark=is_bark,
                confidence=confidence,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error=None,
            )
        except Exception as e:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error=f"Inference error: {e}",
            )

    def export_coreml(self, output_path: str) -> str:
        """モデルを CoreML 形式 (.mlmodel) にエクスポートする。

        Args:
            output_path: エクスポート先のファイルパス。

        Returns:
            エクスポートされたファイルのパス。

        Raises:
            ModelLoadError: モデルが読み込まれていない場合。
            FileNotFoundError: モデルファイルが存在しない場合。
            RuntimeError: 変換に失敗した場合。
        """
        if self._session is None:
            raise ModelLoadError("No model loaded")

        from bark_check.coreml_exporter import CoreMLExporter

        exporter = CoreMLExporter()
        return exporter.export(self._model_path, output_path)
