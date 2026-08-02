# 学習パイプライン

ESC-50 データセットを使って犬の吠え声（bark/howl）検出モデルを学習し、ONNX 形式でエクスポートする。

## 前提条件

- Python 3.12
- `pip install -e ".[train]"` で学習用依存（PyTorch）をインストール済みであること

## クイックスタート

```bash
# 学習用依存のインストール
pip install -e ".[train]"

# 学習実行（ESC-50 の自動ダウンロード → 学習 → ONNX エクスポート）
python -m training
```

以上で `models/bark_model.onnx` が生成される。

## 実行の流れ

1. ESC-50 データセットを GitHub からダウンロード・展開（`data/ESC-50-master/`）
2. 正例（dog クラス）と負例（coughing, sneezing 等）のデータセットを構築
3. BarkCNN（デフォルト）または BarkCNN2d を学習
4. ベストモデルを ONNX 形式でエクスポート
5. ONNX Runtime でサニティチェック

## オプション

```bash
python -m training --help

# モデルタイプを指定（conv1d / conv2d）
python -m training --model-type conv2d

# エポック数を変更
python -m training --epochs 100

# バッチサイズと学習率を変更
python -m training --batch-size 16 --lr 0.0005

# 出力先を変更
python -m training --output models/my_model.onnx

# データディレクトリを変更
python -m training --data-dir /path/to/ESC-50-master

# データソースを指定（ESC-50 + UrbanSound8K）
python -m training --data-sources esc50,urbansound8k

# ESC-50 のみ（デフォルト動作と同じ）
python -m training --data-sources esc50
```

### `--model-type`

| 値 | モデル | 入力形状 | 用途 |
|---|---|---|---|
| `conv1d`（デフォルト） | BarkCNN | `[B, T, 40]` (可変長) | CLI 推論向け |
| `conv2d` | BarkCNN2d | `[B, 1, 40, 199]` (固定長 4D) | CoreML 変換向け |

`conv2d` を選択した場合、ONNX エクスポート後に CoreML 互換性検証が自動実行される。

## データセット

### ESC-50

- URL: https://github.com/karolpiczak/ESC-50
- 50 クラス × 40 クリップ = 2,000 クリップ（各 5 秒、44.1kHz WAV）
- 5-fold cross-validation 構成

### UrbanSound8K

- URL: https://urbansounddataset.weebly.com/urbansound8k.html
- 10 クラス × 10-fold = 8,732 クリップ（各 4 秒以下、WAV）
- ライセンス: Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0)

**重要:** UrbanSound8K はライセンスの制約により自動ダウンロードを行いません。手動でダウンロードしてください。

#### ダウンロード手順

1. [UrbanSound8K ダウンロードページ](https://urbansounddataset.weebly.com/urbansound8k.html) にアクセス
2. データセットをダウンロードして以下の構成で配置:

```
data/UrbanSound8K/
├── metadata/
│   └── UrbanSound8K.csv
└── audio/
    ├── fold1/
    ├── fold2/
    ├── ...
    └── fold10/
```

#### 使用クラス（デフォルト）

**正例 (label=1):**
| classID | クラス名 | 説明 |
|---|---|---|
| 3 | dog_bark | 犬の吠え声 |

**負例 (label=0):**
| classID | クラス名 | 説明 |
|---|---|---|
| 1 | car_horn | クラクション |
| 2 | children_playing | 子供の遊び声 |
| 5 | engine_idling | エンジンのアイドリング音 |
| 8 | siren | サイレン |

### クラス構成（ESC-50）

**正例 (label=1):**
| クラス番号 | カテゴリ | 説明 |
|---|---|---|
| 0 | dog | 犬の吠え声・遠吠え |

**負例 (label=0):**
| クラス番号 | カテゴリ | 説明 |
|---|---|---|
| 5 | cat | 猫の鳴き声 |
| 20 | crying_baby | 赤ちゃんの泣き声 |
| 21 | sneezing | くしゃみ |
| 22 | clapping | 拍手 |
| 23 | breathing | 呼吸音 |
| 24 | coughing | 咳 |
| 26 | laughing | 笑い声 |
| 30 | door_wood_knock | ドアノック |

## モデルアーキテクチャ

### BarkCNN（デフォルト: `--model-type conv1d`）

軽量 1D CNN。可変長入力に対応し、動的軸で ONNX エクスポートされる。

```
入力: [batch, T, 40] (MFCC 特徴量)
  ↓ permute → [batch, 40, T]
Conv1d(40, 64, k=3) → BN → ReLU → MaxPool(2)
Conv1d(64, 128, k=3) → BN → ReLU → MaxPool(2)
Conv1d(128, 128, k=3) → BN → ReLU
  ↓ Global Average Pooling
Linear(128, 1) → Sigmoid
出力: [batch, 1] (吠え声確率)
```

### BarkCNN2d（`--model-type conv2d`）

CoreML 互換の Conv2d ベースモデル。固定長入力・全軸静的で CoreML 変換に適している。

```
入力: [batch, 1, 40, 199] (4D, MFCC 特徴量)
  ↓ permute → [batch, 40, 1, 199]
Conv2d(40, 64, k=(1,3)) → BN2d → ReLU → MaxPool2d((1,2))
Conv2d(64, 128, k=(1,3)) → BN2d → ReLU → MaxPool2d((1,2))
Conv2d(128, 128, k=(1,3)) → BN2d → ReLU
  ↓ AdaptiveAvgPool2d((1,1))
  ↓ Dropout
Linear(128, 1) → Sigmoid
出力: [batch, 1] (吠え声確率)
```

## 特徴量抽出

推論時と同一の `FeatureExtractor` を使用（train/inference skew を防止）:

- MFCC 40 次元
- サンプリングレート: 16,000 Hz
- フレームサイズ: 400 サンプル (25ms)
- フレームシフト: 160 サンプル (10ms)

## カスタマイズ

`training/config.py` の `TrainingConfig` を編集することで以下を変更可能:

- 正例・負例クラスの追加・変更
- バリデーション fold の変更
- ハイパーパラメータの調整
- クロップ長の変更（デフォルト: 2 秒）

## ディレクトリ構成

```
training/
├── __init__.py              # パッケージ初期化
├── __main__.py              # エントリポイント
├── augmentation.py          # データ拡張関数
├── config.py                # ハイパーパラメータ設定
├── dataset.py               # ESC50BarkDataset
├── dataset_base.py          # BarkDatasetBase 抽象基底クラス
├── dataset_factory.py       # DatasetFactory
├── dataset_urbansound8k.py  # UrbanSound8KBarkDataset
├── model.py                 # BarkCNN / BarkCNN2d モデル定義
├── onnx_validator.py        # ONNX CoreML 互換性検証
├── train.py                 # 学習ループ + ONNX エクスポート
└── README.md                # このファイル
```
