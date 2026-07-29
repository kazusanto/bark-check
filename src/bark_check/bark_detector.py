"""犬の吠え声を検出するコアロジックモジュール。"""

from __future__ import annotations

import time

import numpy as np

from bark_check.feature_extractor import FeatureExtractor
from bark_check.models import DetectionResult

# 入力 PCM の最大時間長（秒）
_MAX_DURATION_SEC = 10.0


class ModelLoadError(Exception):
    """モデルの読み込みに失敗した場合に送出される例外。"""


class BarkDetector:
    """犬の吠え声を検出するコアライブラリ。"""

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
        self._fixed_length: int | None = None
        self._channels_first: bool = False
        self._is_4d: bool = False
        self._model_format_error: str | None = None

        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """ONNX モデルを読み込み、入力形状からモデル形式を判別する。

        判別ロジック:
        - input shape が [1, 40, N] → 固定長 channels-first モデル
        - input shape が [1, T, 40] → 可変長 channels-last モデル
        - それ以外 → self._model_format_error に格納

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

        # 入力 shape メタデータからモデル形式を自動判別する
        input_meta = self._session.get_inputs()[0]
        shape = input_meta.shape  # e.g. [1, 1, 40, 199] or [1, 40, 199] or [1, 'T', 40]

        if len(shape) == 4:
            if shape[1] == 1 and shape[2] == 40:
                # 4D: [1, 1, 40, N] (CoreML 互換、固定長)
                self._fixed_length = shape[3] if isinstance(shape[3], int) else 199
                self._channels_first = True
                self._is_4d = True
            else:
                self._model_format_error = (
                    f"Unsupported 4D model format: expected [1, 1, 40, N], got {shape}"
                )
        elif len(shape) == 3:
            if shape[1] == 40:
                # 3D channels-first: [1, 40, N] (固定長)
                self._fixed_length = shape[2] if isinstance(shape[2], int) else 199
                self._channels_first = True
            elif shape[2] == 40:
                # 3D channels-last: [1, T, 40] (可変長)
                self._fixed_length = None
                self._channels_first = False
            else:
                self._model_format_error = (
                    "Unsupported model format: cannot determine channel position"
                )
        else:
            self._model_format_error = (
                f"Unsupported model format: input has {len(shape)} dimensions, expected 3 or 4"
            )

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
        if len(pcm) / sample_rate > _MAX_DURATION_SEC:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error="Input exceeds maximum duration of 10.0 seconds",
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

        # モデル形式エラーチェック
        if self._model_format_error is not None:
            return DetectionResult(
                is_bark=False,
                confidence=0.0,
                timestamp=timestamp,
                audio_duration=audio_duration,
                error=self._model_format_error,
            )

        # 推論
        try:
            if self._channels_first:
                # 固定長 channels-first パス
                features = self._feature_extractor.extract(
                    pcm, sample_rate, fixed_length=self._fixed_length
                )
                # [199, 40] → [40, 199] → [1, 40, 199]
                input_tensor = features.T[np.newaxis, :, :].astype(np.float32)

                if self._is_4d:
                    # [1, 40, 199] → [1, 1, 40, 199]
                    input_tensor = input_tensor[:, np.newaxis, :, :]
            else:
                # 可変長 channels-last パス
                features = self._feature_extractor.extract(pcm, sample_rate)
                # [T, 40] → [1, T, 40]
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


