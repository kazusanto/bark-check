"""バグ条件探索プロパティテスト。

修正前コードでバグの存在を実証するための探索テスト。
サンプル数 > 32,000 かつ時間長 <= 10 秒の入力が不正にエラーを返すことを検出する。

このテストは修正前コードで FAIL することが期待される。
FAIL = バグの存在が確認された（探索成功）。
修正後コードでは PASS する（バグが修正された証拠）。
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from bark_check.bark_detector import BarkDetector


@given(
    sample_rate=st.integers(min_value=8000, max_value=48000),
    duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_bug_condition_no_input_length_error(sample_rate: int, duration: float) -> None:
    """バグ条件に該当する入力が入力長エラーを返さないことを検証する。

    **Validates: Requirements 1.1, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 5.1**

    Bug Condition: len(pcm) > 32000 AND len(pcm) / sample_rate <= 10.0

    期待動作（修正後）: 入力長に関するエラーを返さず、推論処理を実行する。
    修正前コードでは「Input exceeds maximum length of 32000 samples」エラーが
    返されるため、このテストは FAIL する。
    """
    # サンプル数を計算
    num_samples = int(sample_rate * duration)

    # Bug Condition: サンプル数 > 32,000 かつ時間長 <= 10 秒
    assume(num_samples > 32000)
    assume(num_samples / sample_rate <= 10.0)

    # 無音ではない PCM データを生成（無音チェックを通過させるため）
    pcm = np.ones(num_samples, dtype=np.float32) * 0.1

    # モデル未ロード状態で検出を実行
    detector = BarkDetector()
    result = detector.detect(pcm, sample_rate)

    # 入力長に関するエラーが返されないことを確認
    # モデル未ロードエラー ("No model loaded") は許容される
    assert result.error is None or "maximum length" not in result.error, (
        f"Bug detected: input with {num_samples} samples at {sample_rate}Hz "
        f"(duration={num_samples/sample_rate:.2f}s <= 10.0s) was incorrectly rejected "
        f"with error: {result.error}"
    )
