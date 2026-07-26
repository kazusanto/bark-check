"""CLI エントリポイントモジュール。

bark-check コマンドの引数解析、エラー処理、終了コード管理を行う。
"""

from __future__ import annotations

import sys
import argparse

from bark_check.audio_loader import AudioLoader, UnsupportedFormatError, AudioLoadError
from bark_check.bark_detector import BarkDetector, ModelLoadError
from bark_check.output_formatter import OutputFormatter


def main() -> None:
    """bark-check CLI のメインエントリポイント。

    argparse による引数解析を行い、AudioLoader → BarkDetector → OutputFormatter
    のパイプラインで音声ファイルの吠え声判定を実行する。

    終了コード:
        0: 吠え声あり
        1: 入力エラー/推論エラー
        2: 非対応フォーマット
        3: 吠え声なし
        4: モデル読み込み失敗
    """
    parser = argparse.ArgumentParser(
        prog="bark-check",
        description="音声ファイルから犬の吠え声を検出する。",
        epilog="対応フォーマット: wav, mp3, flac, ogg",
    )
    parser.add_argument(
        "audio_file",
        help="判定対象の音声ファイルパス",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="JSON 形式で出力する",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="判定閾値 (0.0〜1.0、デフォルト: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="モデルファイルパス (オプション)",
    )

    args = parser.parse_args()

    # 閾値バリデーション
    if not (0.0 <= args.threshold <= 1.0):
        print(
            f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}",
            file=sys.stderr,
        )
        sys.exit(1)

    # BarkDetector 初期化
    try:
        detector = BarkDetector(
            threshold=args.threshold,
            model_path=args.model,
        )
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(4)

    # 音声ファイル読み込み
    loader = AudioLoader()
    try:
        pcm, sample_rate = loader.load(args.audio_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    except UnsupportedFormatError:
        print(
            "Error: Unsupported format. Supported: wav, mp3, flac, ogg",
            file=sys.stderr,
        )
        sys.exit(2)
    except AudioLoadError as e:
        print(f"Error: Failed to load audio file: {e}", file=sys.stderr)
        sys.exit(1)

    # 吠え声検出
    result = detector.detect(pcm, sample_rate)

    # 推論エラーチェック
    if result.error is not None:
        print(f"Error: Detection failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    # 出力フォーマット
    formatter = OutputFormatter()
    if args.json:
        output = formatter.format_json(result)
    else:
        output = formatter.format_text(result)

    print(output)

    # 終了コード: 吠え声あり=0、なし=3
    if result.is_bark:
        sys.exit(0)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
