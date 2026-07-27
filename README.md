# bark-check

音声ファイルが犬の吠え声を含むかどうかを判定する Python CLI ツール。

判定ロジックはコアライブラリ（`BarkDetector`）として独立しており、iOS Swift アプリなど他のプラットフォームへの移植を容易にする設計になっている。

## インストール

```bash
# Python 3.12 が必要
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 使い方

```bash
# 基本的な使い方
bark-check audio.wav

# JSON 形式で出力
bark-check --json audio.wav

# 判定閾値を変更（デフォルト: 0.5）
bark-check --threshold 0.7 audio.wav

# 学習済みモデルを指定
bark-check --model models/bark_model.onnx audio.wav
```

### 対応フォーマット

WAV, MP3, FLAC, OGG

### 終了コード

| コード | 意味 |
|---|---|
| 0 | 吠え声あり |
| 1 | 入力エラー / 推論エラー |
| 2 | 非対応フォーマット |
| 3 | 吠え声なし |
| 4 | モデル読み込み失敗 |

## アーキテクチャ

```
CLI Layer                          Core Library
┌────────────────────┐             ┌─────────────────────────┐
│ main.py            │             │ bark_detector.py        │
│ audio_loader.py    │──PCM──────▶│ feature_extractor.py    │
│ output_formatter.py│◀─Result────│ models.py               │
└────────────────────┘             └─────────────────────────┘
```

- **BarkDetector**: メモリ上のモノラル PCM（float32）を受け取り、吠え声判定を返すコアモジュール。ファイル I/O を一切持たない。
- **AudioLoader**: 音声ファイルをデコードしてモノラル PCM に変換する CLI レイヤーのモジュール。
- **FeatureExtractor**: 16kHz PCM から MFCC 特徴量（40次元）を抽出する。
- **DetectionResult**: 判定結果を表す dataclass。JSON シリアライズ/デシリアライズ対応。

## ライブラリとして使う

```python
import numpy as np
from bark_check.bark_detector import BarkDetector

detector = BarkDetector(threshold=0.5, model_path="models/bark_model.onnx")

# float32 モノラル PCM + サンプリングレート
pcm = np.random.randn(16000).astype(np.float32)
result = detector.detect(pcm, sample_rate=16000)

print(result.is_bark)      # True / False
print(result.confidence)   # 0.0〜1.0
print(result.error)        # None or エラーメッセージ
```

## モデルの学習・生成

推論に必要な ONNX モデルは、プロジェクト内の学習パイプラインで生成できる。

```bash
# 学習用依存のインストール
pip install -e ".[train]"

# モデル学習（ESC-50 自動ダウンロード → 学習 → ONNX エクスポート）
python -m training
```

実行すると `models/bark_model.onnx` が生成される。詳細は [training/README.md](training/README.md) を参照。

## CoreML エクスポート

iOS / macOS 向けに ONNX モデルを CoreML 形式に変換できる。

```python
detector = BarkDetector(model_path="models/bark_model.onnx")
path = detector.export_coreml("models/bark_model.mlmodel")
```

## テスト

```bash
# 全テスト実行
pytest tests/ -v

# プロパティベーステストのみ
pytest tests/ -v -k "property"
```

## 開発

```bash
# 開発用依存含めてインストール
pip install -e ".[dev]"

# テスト実行
pytest tests/ -v
```

## ディレクトリ構成

```
bark-check/
├── src/bark_check/
│   ├── bark_detector.py       # コアロジック
│   ├── feature_extractor.py   # MFCC 特徴量抽出
│   ├── models.py              # DetectionResult
│   ├── audio_loader.py        # 音声ファイルデコード（CLI）
│   ├── output_formatter.py    # 出力フォーマット（CLI）
│   ├── coreml_exporter.py     # CoreML エクスポート
│   └── main.py                # CLI エントリポイント
├── training/                  # 学習パイプライン
│   ├── config.py              # ハイパーパラメータ設定
│   ├── dataset.py             # ESC-50 Dataset
│   ├── model.py               # BarkCNN モデル定義
│   └── train.py               # 学習 + ONNX エクスポート
├── tests/                     # pytest + hypothesis
├── models/                    # 学習済みモデル置き場
└── pyproject.toml
```

## ライセンス

MIT
