"""保存プロパティテスト（Preservation Property Tests）。

非バグ条件の入力に対する現行コードの動作をベースラインとして記録し、
修正後もこれらの動作が保持されることを検証するプロパティベーステスト。

非バグ条件の入力ドメイン:
- 空入力（サンプル数 0）
- 無音入力（全ゼロ、32,000 サンプル以下）
- 短い入力（サンプル数 <= 32,000、非ゼロ）
- 閾値バリデーション（[0.0, 1.0] 範囲外）

修正前コードで実行 → 全テスト PASS（保存すべき動作のベースラインを確認）
修正後コードで実行 → 全テスト PASS（リグレッションなしの確認）
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bark_check.bark_detector import BarkDetector


# --- Property: 空入力は常に "Input PCM block is empty" エラーを返す ---


@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
)
@settings(max_examples=50)
def test_preservation_empty_input_returns_empty_error(sample_rate: int) -> None:
    """空入力プロパティ: 空配列は常に error="Input PCM block is empty" を返す。

    **Validates: Requirements 4.1, 5.2**

    非バグ条件: len(pcm) == 0 の入力は修正前後で同一動作を維持する。
    """
    detector = BarkDetector()
    pcm = np.array([], dtype=np.float32)
    result = detector.detect(pcm, sample_rate)

    assert result.is_bark is False
    assert result.confidence == 0.0
    assert result.error == "Input PCM block is empty"


# --- Property: 無音入力（全ゼロ、1〜32,000 サンプル）は is_bark=False, confidence=0.0, error=None ---


@given(
    length=st.integers(min_value=1, max_value=32000),
    sample_rate=st.integers(min_value=8000, max_value=48000),
)
@settings(max_examples=100)
def test_preservation_silent_input_returns_no_bark(
    length: int, sample_rate: int
) -> None:
    """無音入力プロパティ: 全ゼロ配列（1〜32,000 サンプル）は常に is_bark=False, confidence=0.0, error=None を返す。

    **Validates: Requirements 4.2, 5.2**

    非バグ条件: 無音入力（全ゼロ）かつ len(pcm) <= 32,000 は修正前後で同一動作を維持する。
    """
    detector = BarkDetector()
    pcm = np.zeros(length, dtype=np.float32)
    result = detector.detect(pcm, sample_rate)

    assert result.is_bark is False
    assert result.confidence == 0.0
    assert result.error is None


# --- Property: 短い入力（len(pcm) <= 32000、非ゼロ）は入力長エラーを返さない ---


@given(
    length=st.integers(min_value=1, max_value=32000),
    sample_rate=st.integers(min_value=8000, max_value=48000),
)
@settings(max_examples=100)
def test_preservation_short_input_no_length_error(
    length: int, sample_rate: int
) -> None:
    """短い入力プロパティ: len(pcm) <= 32,000 の非ゼロ入力は入力長エラーを返さない。

    **Validates: Requirements 4.3, 4.6, 5.2**

    非バグ条件: サンプル数が 32,000 以下の入力は修正前後で入力長エラーにならない。
    モデル未ロード時は "No model loaded" エラーが返るが、入力長に関するエラーではない。
    """
    detector = BarkDetector()
    # 非ゼロの PCM データを生成（無音チェックを通過させる）
    pcm = np.ones(length, dtype=np.float32) * 0.1
    result = detector.detect(pcm, sample_rate)

    # 入力長に関するエラーが返されないことを確認
    assert result.error is None or "maximum length" not in result.error
    assert result.error is None or "maximum duration" not in result.error

    # モデル未ロード時は "No model loaded" エラーが返る（これは正常動作）
    assert result.error == "No model loaded"


# --- Property: 閾値バリデーション（範囲外の threshold → ValueError が送出される） ---


@given(
    threshold=st.one_of(
        st.floats(min_value=-1e6, max_value=-0.001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.001, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=100)
def test_preservation_invalid_threshold_raises_value_error(
    threshold: float,
) -> None:
    """閾値バリデーションプロパティ: 範囲外の threshold は ValueError を送出する。

    **Validates: Requirements 4.5, 5.2**

    非バグ条件: threshold が [0.0, 1.0] 範囲外の場合、修正前後で ValueError が送出される。
    """
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        BarkDetector(threshold=threshold)
