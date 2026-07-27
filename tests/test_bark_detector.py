"""BarkDetector のユニットテストおよびプロパティベーステスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bark_check.bark_detector import BarkDetector, ModelLoadError
from bark_check.models import DetectionResult


# ---- ユニットテスト ----


class TestBarkDetectorInit:
    """BarkDetector の初期化に関するテスト。"""

    def test_デフォルト閾値が0_5であること(self):
        """BarkDetector をデフォルト引数で初期化したとき、閾値が 0.5 であること。"""
        detector = BarkDetector()
        assert detector._threshold == 0.5

    def test_閾値を指定して初期化できること(self):
        """threshold=0.7 を指定して初期化したとき、閾値が 0.7 になること。"""
        detector = BarkDetector(threshold=0.7)
        assert detector._threshold == 0.7

    def test_閾値が範囲外のとき_ValueError_が発生すること(self):
        """threshold が 1.0 を超えた値を指定したとき ValueError が発生すること。"""
        with pytest.raises(ValueError):
            BarkDetector(threshold=1.1)

    def test_閾値が負のとき_ValueError_が発生すること(self):
        """threshold が 0.0 未満の値を指定したとき ValueError が発生すること。"""
        with pytest.raises(ValueError):
            BarkDetector(threshold=-0.1)

    def test_model_path_None_でモデル未ロード状態で初期化されること(self):
        """model_path=None でセッションが None のまま初期化されること。"""
        detector = BarkDetector(model_path=None)
        assert detector._session is None


class TestBarkDetectorDetect:
    """BarkDetector.detect() のユニットテスト。"""

    def test_空入力のとき_error_が返ること(self):
        """空の PCM 配列を渡したとき error='Input PCM block is empty' の DetectionResult が返ること。"""
        detector = BarkDetector()
        pcm = np.array([], dtype=np.float32)
        result = detector.detect(pcm, sample_rate=16000)
        assert result.error == "Input PCM block is empty"
        assert isinstance(result, DetectionResult)

    def test_無音入力で_confidence_が_0_0_になること(self):
        """全ゼロの PCM 配列を渡したとき confidence=0.0 かつ is_bark=False が返ること。"""
        detector = BarkDetector()
        pcm = np.zeros(1000, dtype=np.float32)
        result = detector.detect(pcm, sample_rate=16000)
        assert result.confidence == 0.0
        assert result.is_bark is False
        assert result.error is None

    def test_モデル未ロードのとき_error_が返ること(self):
        """model_path=None の状態で detect() を呼んだとき error='No model loaded' が返ること。"""
        detector = BarkDetector(model_path=None)
        pcm = np.random.rand(1000).astype(np.float32)
        result = detector.detect(pcm, sample_rate=16000)
        assert result.error == "No model loaded"

    def test_上限超過入力のとき_error_が返ること(self):
        """10 秒超の PCM 配列を渡したとき時間長ベースのエラーが返ること。"""
        detector = BarkDetector()
        # 16kHz × 11 秒 = 176,000 サンプル（10 秒超）
        pcm = np.ones(176000, dtype=np.float32) * 0.1
        result = detector.detect(pcm, sample_rate=16000)
        assert result.error == "Input exceeds maximum duration of 10.0 seconds"
        assert result.is_bark is False
        assert result.confidence == 0.0

    def test_44_1kHz_10秒以内の入力がエラーなく処理されること(self):
        """44.1kHz で 5 秒の入力（220,500 サンプル > 32,000）が入力長エラーにならないこと。"""
        detector = BarkDetector()
        # 44100Hz × 5 秒 = 220,500 サンプル
        pcm = np.ones(220500, dtype=np.float32) * 0.1
        result = detector.detect(pcm, sample_rate=44100)
        # モデル未ロードなので "No model loaded" エラーが返るが、入力長エラーではない
        assert result.error == "No model loaded"

    def test_ちょうど10秒の入力がエラーにならないこと(self):
        """ちょうど 10 秒の入力（境界値）が入力長エラーにならないこと。"""
        detector = BarkDetector()
        # 16000Hz × 10 秒 = 160,000 サンプル（duration == 10.0）
        pcm = np.ones(160000, dtype=np.float32) * 0.1
        result = detector.detect(pcm, sample_rate=16000)
        # モデル未ロードなので "No model loaded" エラーが返るが、入力長エラーではない
        assert result.error == "No model loaded"

    def test_10秒をわずかに超える入力がエラーになること(self):
        """10 秒をわずかに超える入力（境界値）が入力長エラーになること。"""
        detector = BarkDetector()
        # 16000Hz × 10 秒 + 1 サンプル = 160,001 サンプル（duration > 10.0）
        pcm = np.ones(160001, dtype=np.float32) * 0.1
        result = detector.detect(pcm, sample_rate=16000)
        assert result.error == "Input exceeds maximum duration of 10.0 seconds"
        assert result.is_bark is False
        assert result.confidence == 0.0

    def test_detect_は_DetectionResult_を返すこと(self):
        """detect() の戻り値が DetectionResult のインスタンスであること。"""
        detector = BarkDetector()
        pcm = np.zeros(100, dtype=np.float32)
        result = detector.detect(pcm, sample_rate=16000)
        assert isinstance(result, DetectionResult)


# ---- プロパティベーステスト ----


# Feature: bark-check, Property 1: 有効な PCM 入力は常に有効な DetectionResult を返す
@given(
    pcm=st.lists(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32000,
    ).map(lambda x: np.array(x, dtype=np.float32)),
    sample_rate=st.integers(min_value=8000, max_value=48000),
)
@settings(max_examples=100)
def test_property_valid_pcm_returns_valid_result(
    pcm: np.ndarray, sample_rate: int
) -> None:
    """Validates: Requirements 2.1, 5.4

    1〜32000 サンプルの float32 配列を入力したとき、is_bark が bool、
    confidence が [0.0, 1.0]、error フィールドが存在する DetectionResult を返すこと。
    model_path=None なので 'No model loaded' エラーが返るが、DetectionResult は有効。
    """
    detector = BarkDetector()
    result = detector.detect(pcm, sample_rate)

    assert isinstance(result, DetectionResult)
    assert isinstance(result.is_bark, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert hasattr(result, "error")
    # error は None か非空文字列であること
    assert result.error is None or (isinstance(result.error, str) and len(result.error) > 0)


# Feature: bark-check, Property 2: 閾値による吠え声判定の一貫性
@given(
    confidence=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    threshold=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_property_threshold_consistency(confidence: float, threshold: float) -> None:
    """Validates: Requirements 2.2, 2.3

    モデルありの場合: confidence >= threshold なら is_bark=True、< threshold なら False。
    onnxruntime のモックで confidence を直接制御してテスト。
    """
    detector = BarkDetector(threshold=threshold)

    # セッションをモックして任意の confidence を返すようにする
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [np.array([[confidence]], dtype=np.float32)]
    detector._session = mock_session

    # 無音・全ゼロではない PCM を用意する（無音チェックを通過させる）
    pcm = np.ones(1000, dtype=np.float32) * 0.1
    result = detector.detect(pcm, sample_rate=16000)

    # float32 変換後の実際の confidence 値で期待値を計算する
    actual_confidence = float(np.clip(np.float32(confidence), 0.0, 1.0))
    expected_is_bark = actual_confidence >= threshold
    assert result.is_bark == expected_is_bark
    assert abs(result.confidence - actual_confidence) < 1e-6
    assert result.error is None


# Feature: bark-check, Property 3: 時間長超過入力はエラー DetectionResult を返す
@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
    duration=st.floats(min_value=10.01, max_value=30.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_oversized_input_returns_error(sample_rate: int, duration: float) -> None:
    """Validates: Requirements 1.2, 2.1, 5.3

    時間長が 10 秒を超える入力を渡したとき、
    error="Input exceeds maximum duration of 10.0 seconds",
    is_bark=False, confidence=0.0 の DetectionResult が返ること。
    """
    num_samples = int(sample_rate * duration)
    detector = BarkDetector()
    pcm = np.ones(num_samples, dtype=np.float32) * 0.1
    result = detector.detect(pcm, sample_rate)

    assert isinstance(result, DetectionResult)
    assert result.error == "Input exceeds maximum duration of 10.0 seconds"
    assert result.is_bark is False
    assert result.confidence == 0.0


# Feature: bark-check, Property 6: 無音入力は常に confidence 0.0 の吠え声なしを返す
@given(
    length=st.integers(min_value=1, max_value=32000),
    sample_rate=st.integers(min_value=8000, max_value=48000),
)
@settings(max_examples=100)
def test_property_silent_input_returns_no_bark(length: int, sample_rate: int) -> None:
    """Validates: Requirements 5.3

    全ゼロ配列を渡したとき is_bark=False, confidence=0.0 が返ること。
    """
    detector = BarkDetector()
    pcm = np.zeros(length, dtype=np.float32)
    result = detector.detect(pcm, sample_rate)

    assert result.is_bark is False
    assert result.confidence == 0.0


# Feature: bark-check, Property 7: 推論エラーは DetectionResult に格納され例外は伝播しない
@given(
    error_message=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=200,
    ),
)
@settings(max_examples=100)
def test_property_inference_error_stored_in_result(error_message: str) -> None:
    """Validates: Requirements 5.1, 5.6

    unittest.mock で FeatureExtractor.extract が例外を送出するようにモックし、
    detect() が例外を伝播せず error フィールドに格納して返すことを確認。
    """
    detector = BarkDetector()

    # モデルがロードされている状態に見せかける
    mock_session = MagicMock()
    detector._session = mock_session

    # FeatureExtractor.extract が例外を送出するようにモックする
    with patch.object(
        detector._feature_extractor,
        "extract",
        side_effect=RuntimeError(error_message),
    ):
        pcm = np.ones(1000, dtype=np.float32) * 0.1
        try:
            result = detector.detect(pcm, sample_rate=16000)
        except Exception as e:
            pytest.fail(f"detect() が例外を伝播した: {e}")

    assert isinstance(result, DetectionResult)
    assert result.error is not None
    assert "Inference error:" in result.error
    assert result.is_bark is False


# ---- CoreML エクスポート テスト ----


class TestBarkDetectorExportCoreml:
    """BarkDetector.export_coreml() のユニットテスト。"""

    def test_モデル未ロード時に_ModelLoadError_が発生すること(self):
        """model_path=None の状態で export_coreml() を呼ぶと ModelLoadError が発生すること。"""
        detector = BarkDetector()
        with pytest.raises(ModelLoadError):
            detector.export_coreml("/tmp/output.mlmodel")

    def test_モデルロード済み時に_CoreMLExporter_が呼ばれること(self):
        """セッションがある状態で export_coreml() を呼ぶと CoreMLExporter.export() が呼ばれること。"""
        detector = BarkDetector()
        detector._session = MagicMock()
        detector._model_path = "/tmp/dummy_model.onnx"

        with patch("bark_check.coreml_exporter.CoreMLExporter") as MockExporter:
            mock_instance = MockExporter.return_value
            mock_instance.export.return_value = "/tmp/output.mlmodel"

            result = detector.export_coreml("/tmp/output.mlmodel")

        mock_instance.export.assert_called_once_with("/tmp/dummy_model.onnx", "/tmp/output.mlmodel")
        assert result == "/tmp/output.mlmodel"
