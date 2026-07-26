"""CLI (main.py) のプロパティベーステストおよびユニットテスト。"""

from __future__ import annotations

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st

from bark_check.models import DetectionResult


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """CLIを実行して(exit_code, stdout, stderr)を返すヘルパー。"""
    captured_out = StringIO()
    captured_err = StringIO()
    exit_code = 0
    with patch("sys.argv", ["bark-check"] + args), \
         patch("sys.stdout", captured_out), \
         patch("sys.stderr", captured_err):
        try:
            from bark_check.main import main
            main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0
    return exit_code, captured_out.getvalue(), captured_err.getvalue()


# ---- プロパティベーステスト ----


# 非対応拡張子生成戦略: wav, mp3, flac, ogg 以外の拡張子を生成
_unsupported_extensions = st.sampled_from([
    ".txt", ".pdf", ".doc", ".png", ".jpg", ".csv", ".xml", ".html",
    ".py", ".rs", ".go", ".aac", ".wma", ".m4a", ".aiff",
])


# Feature: bark-check, Property 4: 無効拡張子は常に終了コード 2 で拒否される
# Validates: Requirements 1.3
@given(ext=_unsupported_extensions)
@settings(max_examples=50)
def test_property_invalid_extension_exits_with_code_2(ext: str) -> None:
    """非対応拡張子のパスで CLI を呼ぶと exit code 2 で終了すること。

    Validates: Requirements 1.3
    """
    # 一時ファイルを作成して実ファイルとして存在させる
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(b"dummy content")
        tmp_path = f.name

    try:
        exit_code, _, _ = _run_cli([tmp_path])
        assert exit_code == 2, (
            f"Expected exit code 2 for extension {ext}, got {exit_code}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Feature: bark-check, Property 5: --json 出力は常に有効なスキーマを持つ
# Validates: Requirements 3.2
@given(
    confidence=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    is_bark=st.booleans(),
)
@settings(max_examples=50)
def test_property_json_output_has_valid_schema(
    confidence: float, is_bark: bool
) -> None:
    """有効な WAV ファイル + --json で呼ぶと stdout が valid JSON で is_bark と confidence を含むこと。

    Validates: Requirements 3.2
    """
    # 一時 WAV ファイルを生成
    sample_rate = 16000
    pcm_data = np.random.rand(8000).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, pcm_data, sample_rate)

    # detect() をモックして任意の DetectionResult を返す
    mock_result = DetectionResult(
        is_bark=is_bark,
        confidence=confidence,
        timestamp=1000.0,
        audio_duration=0.5,
        error=None,
    )

    try:
        with patch("bark_check.main.BarkDetector") as MockDetector:
            instance = MockDetector.return_value
            instance.detect.return_value = mock_result

            exit_code, stdout, _ = _run_cli(["--json", tmp_path])

        # JSON として有効であること
        parsed = json.loads(stdout)
        assert "is_bark" in parsed
        assert "confidence" in parsed
        assert isinstance(parsed["is_bark"], bool)
        assert isinstance(parsed["confidence"], (int, float))
        assert parsed["is_bark"] == is_bark
        assert abs(parsed["confidence"] - confidence) < 1e-6
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---- ユニットテスト ----


class TestCliHelp:
    """CLI のヘルプ表示テスト。"""

    def test_helpオプションで終了コード0とwavを含む出力(self):
        """--help で終了コード 0 が返り、出力に 'wav' が含まれること。"""
        exit_code, stdout, _ = _run_cli(["--help"])
        assert exit_code == 0
        assert "wav" in stdout.lower()


class TestCliArgValidation:
    """CLI の引数バリデーションテスト。"""

    def test_引数未指定で終了コードが0でないこと(self):
        """引数を指定しなかったとき終了コードが 0 でないこと。"""
        exit_code, _, _ = _run_cli([])
        assert exit_code != 0

    def test_存在しないパスで終了コード1(self):
        """存在しないファイルパスを指定したとき終了コード 1 が返ること。"""
        with patch("bark_check.main.BarkDetector"):
            exit_code, _, _ = _run_cli(["/tmp/nonexistent_file_99999.wav"])
        assert exit_code == 1

    def test_threshold_1_5で終了コード1(self):
        """--threshold 1.5 を指定したとき終了コード 1 が返ること。"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            f.write(b"dummy")

        try:
            with patch("bark_check.main.BarkDetector"):
                exit_code, _, _ = _run_cli(["--threshold", "1.5", tmp_path])
            assert exit_code == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_モデル読み込み失敗で終了コード4(self):
        """存在しないパスを --model に指定したとき終了コード 4 が返ること。"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            f.write(b"dummy")

        try:
            exit_code, _, _ = _run_cli([
                "--model", "/tmp/nonexistent_model_99999.onnx",
                tmp_path,
            ])
            assert exit_code == 4
        finally:
            Path(tmp_path).unlink(missing_ok=True)
