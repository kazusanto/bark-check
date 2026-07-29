"""データ拡張関数モジュール。

学習時に PCM 波形に対して適用するデータ拡張（タイムシフト、ガウシアンノイズ付加）
を提供する。
"""

import numpy as np


def apply_time_shift(
    pcm: np.ndarray,
    max_shift: int = 1600,
) -> np.ndarray:
    """PCM 波形を時間方向にランダムシフトする。

    ±max_shift サンプルの範囲で一様分布に従いランダムにシフトし、
    範囲外となった部分はゼロ埋めする。

    Args:
        pcm: 入力 PCM 配列（float32）。
        max_shift: 最大シフト量（サンプル数）。±max_shift の範囲。

    Returns:
        シフト後の PCM 配列（元と同じ長さ、範囲外はゼロ埋め）。
    """
    if len(pcm) == 0:
        return pcm

    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift == 0:
        return pcm.copy()

    result = np.zeros_like(pcm)

    if shift > 0:
        # 右シフト: 先頭に shift サンプル分のゼロが入り、末尾が切れる
        if shift < len(pcm):
            result[shift:] = pcm[: len(pcm) - shift]
    else:
        # 左シフト: 末尾に |shift| サンプル分のゼロが入り、先頭が切れる
        abs_shift = -shift
        if abs_shift < len(pcm):
            result[: len(pcm) - abs_shift] = pcm[abs_shift:]

    return result


def apply_gaussian_noise(
    pcm: np.ndarray,
    snr_min: float = 20.0,
    snr_max: float = 40.0,
) -> np.ndarray:
    """PCM 波形にガウシアンノイズを付加する。

    SNR を [snr_min, snr_max] dB の範囲で一様分布に従い選択し、
    その SNR 値に対応するガウシアンノイズを加算する。

    Args:
        pcm: 入力 PCM 配列（float32）。
        snr_min: 最小 SNR（dB）。
        snr_max: 最大 SNR（dB）。

    Returns:
        ノイズ付加後の PCM 配列（元と同じ長さ）。
    """
    if len(pcm) == 0:
        return pcm

    # SNR を一様分布で選択
    snr_db = np.random.uniform(snr_min, snr_max)

    # 信号パワーを計算
    signal_power = np.mean(pcm**2)

    # 信号パワーがゼロの場合（無音信号）はそのまま返す
    if signal_power == 0.0:
        return pcm.copy()

    # SNR からノイズパワーを算出
    # SNR(dB) = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / 10^(SNR/10)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)

    # ガウシアンノイズを生成して加算
    noise = np.random.normal(0.0, noise_std, size=pcm.shape).astype(pcm.dtype)

    return pcm + noise
