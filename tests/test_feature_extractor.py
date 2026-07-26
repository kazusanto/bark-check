"""FeatureExtractor のユニットテスト。"""

import numpy as np
import pytest

from bark_check.feature_extractor import FeatureExtractor

# テスト用定数
_SAMPLE_RATE_16K = 16000
_SAMPLE_RATE_8K = 8000
_N_MFCC = 40
_HOP_LENGTH = 160
_N_FFT = 400


def _make_sine_pcm(duration_sec: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    """テスト用の正弦波 PCM を生成する。

    Args:
        duration_sec: 波形の長さ（秒）。
        sample_rate: サンプリングレート（Hz）。
        freq: 正弦波の周波数（Hz）。

    Returns:
        float32 型のモノラル PCM 配列。
    """
    t = np.linspace(0.0, duration_sec, int(duration_sec * sample_rate), endpoint=False)
    return (np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


class TestFeatureExtractorOutputShape:
    """extract() の出力形状に関するテスト。"""

    def test_16kHz_1秒入力の出力形状が_T_40になること(self):
        """16kHz の正弦波 PCM（1秒）を入力したとき出力形状が [T, 40] になること。"""
        extractor = FeatureExtractor()
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_16K)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        assert result.ndim == 2
        assert result.shape[1] == _N_MFCC

    def test_出力フレーム数が期待値と一致すること(self):
        """16kHz・1秒の PCM を入力したとき、フレーム数が計算上の期待値と一致すること。"""
        extractor = FeatureExtractor()
        n_samples = _SAMPLE_RATE_16K  # 1秒分
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_16K)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        # librosa の frame 計算: 1 + (n_samples - n_fft) // hop_length
        # ただし n_fft より短いケースは center=True でパディングされる
        expected_t = 1 + (n_samples) // _HOP_LENGTH  # center=True のデフォルト近似
        # 大まかな範囲で確認（librosa の center パラメータの影響を許容）
        assert abs(result.shape[0] - expected_t) <= 2


class TestFeatureExtractorResampling:
    """リサンプリング処理に関するテスト。"""

    def test_8kHz入力は16kHzリサンプリング後と同等のフレーム数になること(self):
        """8kHz の入力（8000サンプル）を渡したとき、16kHz リサンプリング後と同等のフレーム数になること。"""
        extractor = FeatureExtractor()

        # 8kHz・1秒の PCM（8000サンプル）
        pcm_8k = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_8K)
        result_8k = extractor.extract(pcm_8k, sample_rate=_SAMPLE_RATE_8K)

        # 16kHz・1秒の PCM（16000サンプル）
        pcm_16k = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_16K)
        result_16k = extractor.extract(pcm_16k, sample_rate=_SAMPLE_RATE_16K)

        # リサンプリング後は同じサンプル数になるため、フレーム数も一致すること
        assert result_8k.shape[0] == result_16k.shape[0]
        assert result_8k.shape[1] == _N_MFCC

    def test_8kHz入力の出力形状が_T_40になること(self):
        """8kHz の PCM を入力したとき出力形状が [T, 40] になること。"""
        extractor = FeatureExtractor()
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_8K)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_8K)

        assert result.ndim == 2
        assert result.shape[1] == _N_MFCC


class TestFeatureExtractorDtype:
    """出力データ型に関するテスト。"""

    def test_16kHz入力の出力がfloat32型であること(self):
        """16kHz の入力を渡したとき出力が float32 型であること。"""
        extractor = FeatureExtractor()
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_16K)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        assert result.dtype == np.float32

    def test_float64入力でも出力がfloat32型であること(self):
        """float64 型の PCM を渡したとき出力が float32 型であること。"""
        extractor = FeatureExtractor()
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_16K).astype(np.float64)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        assert result.dtype == np.float32

    def test_8kHz入力の出力がfloat32型であること(self):
        """8kHz の入力を渡したとき出力が float32 型であること。"""
        extractor = FeatureExtractor()
        pcm = _make_sine_pcm(duration_sec=1.0, sample_rate=_SAMPLE_RATE_8K)

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_8K)

        assert result.dtype == np.float32


class TestFeatureExtractorMinimumLength:
    """最小長 PCM に関するテスト。"""

    def test_1フレーム分以上のPCMで正常に動作すること(self):
        """1フレーム分以上の PCM を渡したとき例外なく動作し、出力形状が [T, 40] になること。"""
        extractor = FeatureExtractor()
        # n_fft（400サンプル）より少し長い入力で1フレームが取れることを確認
        n_samples = _N_FFT + _HOP_LENGTH  # 560サンプル = 最低2フレーム相当
        pcm = np.zeros(n_samples, dtype=np.float32)
        pcm[0] = 1.0  # 無音でないように1サンプルだけ値を入れる

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        assert result.ndim == 2
        assert result.shape[0] >= 1
        assert result.shape[1] == _N_MFCC

    def test_n_fft長のPCMでも正常に動作すること(self):
        """n_fft（400）サンプルちょうどの PCM を渡したとき例外なく動作すること。"""
        extractor = FeatureExtractor()
        pcm = np.ones(_N_FFT, dtype=np.float32) * 0.1

        result = extractor.extract(pcm, sample_rate=_SAMPLE_RATE_16K)

        assert result.ndim == 2
        assert result.shape[0] >= 1
        assert result.shape[1] == _N_MFCC
