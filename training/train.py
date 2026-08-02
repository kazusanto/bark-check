"""学習ループと ONNX エクスポートを実行するメインモジュール。"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.dataset import download_esc50
from training.dataset_factory import build_dataset
from training.model import BarkCNN, BarkCNN2d, count_parameters
from training.onnx_validator import validate_onnx_for_coreml


def _collate_fn_variable(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """可変長の MFCC テンソルをゼロパディングでバッチ化する。

    Args:
        batch: (features [T_i, 40], label [1]) のリスト。

    Returns:
        (padded_features [B, T_max, 40], labels [B, 1]) のタプル。
    """
    features_list, labels_list = zip(*batch)

    # 最大フレーム長を取得
    max_len = max(f.shape[0] for f in features_list)

    # ゼロパディング
    padded = []
    for f in features_list:
        pad_size = max_len - f.shape[0]
        if pad_size > 0:
            f = torch.nn.functional.pad(f, (0, 0, 0, pad_size))
        padded.append(f)

    features = torch.stack(padded)
    labels = torch.stack(labels_list)

    return features, labels


def _collate_fn_fixed(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """固定長テンソルをバッチ化する（パディング不要）。

    Args:
        batch: (features [1, 40, 199], label [1]) のリスト。

    Returns:
        (features [B, 1, 40, 199], labels [B, 1]) のタプル。
    """
    features_list, labels_list = zip(*batch)
    features = torch.stack(features_list)
    labels = torch.stack(labels_list)
    return features, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """1 エポック分の学習を実行する。

    Args:
        model: 学習対象モデル。
        dataloader: 学習データローダー。
        criterion: 損失関数。
        optimizer: オプティマイザ。

    Returns:
        エポックの平均損失。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """バリデーションデータで評価する。

    Args:
        model: 評価対象モデル。
        dataloader: バリデーションデータローダー。
        criterion: 損失関数。

    Returns:
        (平均損失, accuracy) のタプル。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def export_onnx(model: nn.Module, config: TrainingConfig) -> None:
    """学習済みモデルを ONNX 形式でエクスポートする。

    Args:
        model: エクスポート対象のモデル。
        config: 学習設定（出力パスを参照）。
    """
    model.eval()

    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)

    if config.model_type == "conv2d":
        # BarkCNN2d: 4D channels-first 固定長入力、全軸静的
        dummy_input = torch.randn(1, 1, 40, config.fixed_frame_length)
        torch.onnx.export(
            model,
            dummy_input,
            str(config.output_model_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=9,
            dynamo=False,
        )
    else:
        # BarkCNN: channels-last 可変長入力、動的軸あり
        dummy_frames = 199
        dummy_input = torch.randn(1, dummy_frames, 40)
        torch.onnx.export(
            model,
            dummy_input,
            str(config.output_model_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "frames"},
                "output": {0: "batch"},
            },
            opset_version=9,
            dynamo=False,
        )

    print(f"ONNX モデルをエクスポートしました: {config.output_model_path}")


def verify_onnx(config: TrainingConfig) -> None:
    """エクスポートした ONNX モデルの推論が動作することを検証する。

    Args:
        config: 学習設定（モデルパスを参照）。
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(config.output_model_path))
    input_name = session.get_inputs()[0].name

    # モデルタイプに応じたダミー入力でサニティチェック
    if config.model_type == "conv2d":
        # BarkCNN2d: [1, 1, 40, 199] (4D channels-first)
        dummy_input = np.random.randn(1, 1, 40, config.fixed_frame_length).astype(np.float32)
    else:
        # BarkCNN: [1, 199, 40] (channels-last)
        dummy_input = np.random.randn(1, 199, 40).astype(np.float32)

    outputs = session.run(None, {input_name: dummy_input})

    result = outputs[0][0][0]
    print(f"ONNX 推論サニティチェック: output={result:.4f} (0.0〜1.0 の範囲であること)")
    assert 0.0 <= result <= 1.0, f"出力が範囲外: {result}"
    print("サニティチェック OK")


def main() -> None:
    """学習パイプラインのメインエントリポイント。

    ESC-50 ダウンロード → データセット構築 → モデル学習 → ONNX エクスポート
    の全工程を実行する。
    """
    parser = argparse.ArgumentParser(
        description="bark-check 学習パイプライン: ESC-50 で犬の吠え声検出モデルを学習する"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="学習エポック数（デフォルト: 50）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="バッチサイズ（デフォルト: 32）"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="学習率（デフォルト: 1e-3）"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="ESC-50 データディレクトリ"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="出力 ONNX モデルパス"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["conv1d", "conv2d"],
        help="モデルタイプ（デフォルト: conv1d）",
    )
    parser.add_argument(
        "--data-sources",
        type=str,
        default=None,
        help="使用するデータソース（カンマ区切り）。例: esc50,urbansound8k",
    )
    args = parser.parse_args()

    # 設定の構築
    config = TrainingConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.data_dir is not None:
        from pathlib import Path

        config.data_dir = Path(args.data_dir)
    if args.output is not None:
        from pathlib import Path

        config.output_model_path = Path(args.output)
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.data_sources is not None:
        config.data_sources = [s.strip() for s in args.data_sources.split(",")]

    print("=" * 60)
    print("bark-check 学習パイプライン")
    print("=" * 60)
    print(f"  エポック数: {config.epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  クロップ長: {config.clip_duration_sec}s")
    print(f"  サンプリングレート: {config.sample_rate} Hz")
    print(f"  モデルタイプ: {config.model_type}")
    print(f"  データソース: {config.data_sources}")
    print(f"  出力先: {config.output_model_path}")
    print("=" * 60)

    # Step 1: データセットの準備
    print("\n[1/5] データセットの準備...")
    for source in config.data_sources:
        if source == "esc50":
            print("  [esc50]")
            print(f"    ディレクトリ: {config.data_dir}")
            print(f"    正例クラス: {config.positive_classes}")
            print(f"    負例クラス: {config.negative_classes}")
            print(f"    バリデーション fold: {config.val_fold}")
            download_esc50(config.data_dir)
        elif source == "urbansound8k":
            print("  [urbansound8k]")
            print(f"    ディレクトリ: {config.urbansound8k_dir}")
            print(f"    正例クラス: {config.urbansound8k_positive_classes}")
            print(f"    負例クラス: {config.urbansound8k_negative_classes}")
            print(f"    バリデーション fold: {config.urbansound8k_val_fold}")

    # Step 2: データセット構築
    print("\n[2/5] データセットの構築...")
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    try:
        train_dataset = build_dataset(config, is_train=True)
        val_dataset = build_dataset(config, is_train=False)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  学習サンプル数: {len(train_dataset)}")
    print(f"  バリデーションサンプル数: {len(val_dataset)}")

    collate_fn = _collate_fn_fixed if config.model_type == "conv2d" else _collate_fn_variable

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Step 3: モデル構築
    print("\n[3/5] モデルの構築...")
    if config.model_type == "conv2d":
        model = BarkCNN2d(dropout_rate=config.dropout_rate)
        model_name = "BarkCNN2d"
    else:
        model = BarkCNN(dropout_rate=config.dropout_rate)
        model_name = "BarkCNN"
    num_params = count_parameters(model)
    print(f"  モデル: {model_name} ({num_params:,} parameters)")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Step 4: 学習
    print("\n[4/5] 学習開始...")
    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            marker = " *"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:3d}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}{marker} | "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print(f"\n  学習完了 (elapsed: {elapsed:.1f}s)")
    print(f"  最良バリデーション精度: {best_val_acc:.4f}")

    # ベストモデルをロード
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Step 5: ONNX エクスポート
    print("\n[5/5] ONNX エクスポート...")
    export_onnx(model, config)
    verify_onnx(config)

    # ONNX CoreML 互換性検証（conv2d）
    if config.model_type == "conv2d":
        errors = validate_onnx_for_coreml(str(config.output_model_path))
        if errors:
            print("\nONNX CoreML 互換性検証: 失敗", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nONNX CoreML 互換性検証: 全項目合格 ✓")

    print("\n" + "=" * 60)
    print("完了！")
    print(f"  モデルファイル: {config.output_model_path}")
    print(f"  使い方: bark-check --model {config.output_model_path} audio.wav")
    print("=" * 60)


if __name__ == "__main__":
    main()
