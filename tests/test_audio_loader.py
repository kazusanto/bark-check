"""AudioLoader のユニットテスト。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bark_check.audio_loader import AudioLoader, UnsupportedFormatError


class TestAudioLoaderLoad:
    """AudioLoader.load() のユニットテスト。"""

    def test_存在しないパスで_FileNotFoundError_が送出されること(self):
        """存在しないファイルパスを指定したとき FileNotFoundError が送出されること。"""
        loader = AudioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/tmp/does_not_exist_12345.wav")

    def test_非対応拡張子で_UnsupportedFormatError_が送出されること(self):
        """拡張子 .txt のファイルを指定したとき UnsupportedFormatError が送出されること。"""
        loader = AudioLoader()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio")
            tmp_path = f.name

        try:
            with pytest.raises(UnsupportedFormatError):
                loader.load(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_正常なWAVファイルでタプルが返ること(self):
        """有効な WAV ファイルを読み込んだとき (pcm, sample_rate) タプルが返ること。"""
        loader = AudioLoader()
        sample_rate = 16000
        duration_sec = 0.5
        samples = int(sample_rate * duration_sec)
        pcm_data = np.random.rand(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            sf.write(tmp_path, pcm_data, sample_rate)
            result = loader.load(tmp_path)

            assert isinstance(result, tuple)
            assert len(result) == 2
            pcm, sr = result
            assert sr == sample_rate
            assert len(pcm) == samples
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_返り値のpcmがfloat32でndim1であること(self):
        """WAV ファイルの読み込み結果が float32 型で 1 次元配列であること。"""
        loader = AudioLoader()
        sample_rate = 16000
        samples = 8000
        pcm_data = np.random.rand(samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            sf.write(tmp_path, pcm_data, sample_rate)
            pcm, _ = loader.load(tmp_path)

            assert pcm.dtype == np.float32
            assert pcm.ndim == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)
