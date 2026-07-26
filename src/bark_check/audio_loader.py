"""音声ファイルの読み込みとデコードを行うモジュール。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class UnsupportedFormatError(Exception):
    """対応外の音声フォーマットの場合に送出される例外。"""


class AudioLoadError(Exception):
    """音声ファイルのデコードに失敗した場合に送出される例外。"""


class AudioLoader:
    """音声ファイルをデコードしてモノラル PCM に変換する CLI レイヤーのモジュール。"""

    SUPPORTED_FORMATS = ("wav", "mp3", "flac", "ogg")

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """音声ファイルをデコードしてモノラル float32 PCM に変換する。

        soundfile でのデコードを試み、失敗した場合は librosa にフォールバックする。
        ステレオ音声はチャンネル平均によりモノラルに変換される。

        Args:
            file_path: 音声ファイルのパス。

        Returns:
            (pcm, sample_rate) のタプル。pcm は float32 配列、sample_rate は Hz。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            UnsupportedFormatError: 対応外の拡張子の場合。
            AudioLoadError: ファイルのデコードに失敗した場合。
        """
        # 1. ファイル存在チェック
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. 拡張子チェック
        ext = Path(file_path).suffix.lstrip(".").lower()
        if ext not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS)
            raise UnsupportedFormatError(
                f"Unsupported format: .{ext}. Supported: {supported}"
            )

        # 3. デコード（soundfile → librosa フォールバック）
        pcm: np.ndarray | None = None
        sample_rate: int | None = None

        try:
            import soundfile as sf

            data, sr = sf.read(file_path, dtype="float32")
            pcm = data
            sample_rate = sr
        except Exception:
            # soundfile 失敗時は librosa にフォールバック
            try:
                import librosa

                data, sr = librosa.load(file_path, sr=None, mono=False)
                pcm = data
                sample_rate = int(sr)
            except Exception as e:
                raise AudioLoadError(
                    f"Failed to decode audio file: {file_path}: {e}"
                ) from e

        # 4. ステレオ → モノラル変換（チャンネル平均）
        if pcm.ndim == 2:
            # soundfile: shape (samples, channels), librosa: shape (channels, samples)
            if pcm.shape[0] > pcm.shape[1]:
                # soundfile 形式: (samples, channels)
                pcm = pcm.mean(axis=1)
            else:
                # librosa 形式: (channels, samples)
                pcm = pcm.mean(axis=0)

        # 5. float32 に変換
        pcm = pcm.astype(np.float32)

        return pcm, sample_rate
