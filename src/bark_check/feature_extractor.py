"""PCM データから機械学習モデルへの入力特徴量を抽出するモジュール。"""

import numpy as np
import librosa


# MFCC 抽出のデフォルトパラメータ
_TARGET_SAMPLE_RATE = 16000  # 16kHz にリサンプリング
_N_MFCC = 40                 # MFCC 係数数
_N_FFT = 400                 # フレームサイズ（25ms @ 16kHz）
_HOP_LENGTH = 160            # フレームシフト（10ms @ 16kHz）


class FeatureExtractor:
    """PCM データから機械学習モデルへの入力特徴量を抽出する。"""

    def extract(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """モノラル PCM から MFCC 特徴量を抽出する。

        入力 PCM を 16,000 Hz にリサンプリングした後、MFCC を計算して返す。
        フレームサイズは 400 サンプル（25ms）、フレームシフトは 160 サンプル（10ms）。

        Args:
            pcm: float32 型のモノラル PCM サンプル配列 (shape: [N])。
            sample_rate: サンプリングレート（Hz）。

        Returns:
            MFCC 特徴量テンソル (shape: [T, 40])。T はフレーム数。
        """
        # float32 に変換して確実に型を統一する
        pcm_f32 = pcm.astype(np.float32)

        # 16kHz へリサンプリング（すでに 16kHz の場合はそのまま）
        if sample_rate != _TARGET_SAMPLE_RATE:
            pcm_f32 = librosa.resample(
                pcm_f32,
                orig_sr=sample_rate,
                target_sr=_TARGET_SAMPLE_RATE,
            )

        # MFCC 抽出: shape は [N_MFCC, T]
        mfcc = librosa.feature.mfcc(
            y=pcm_f32,
            sr=_TARGET_SAMPLE_RATE,
            n_mfcc=_N_MFCC,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
        )

        # 転置して [T, N_MFCC] 形式に変換する
        return mfcc.T.astype(np.float32)
