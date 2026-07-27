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
3. 軽量 1D CNN (BarkCNN) を学習
4. ベストモデルを ONNX 形式でエクスポート
5. ONNX Runtime でサニティチェック

## オプション

```bash
python -m training --help

# エポック数を変更
python -m training --epochs 100

# バッチサイズと学習率を変更
python -m training --batch-size 16 --lr 0.0005

# 出力先を変更
python -m training --output models/my_model.onnx

# データディレクトリを変更
python -m training --data-dir /path/to/ESC-50-master
```

## データセット

### ESC-50

- URL: https://github.com/karolpiczak/ESC-50
- 50 クラス × 40 クリップ = 2,000 クリップ（各 5 秒、44.1kHz WAV）
- 5-fold cross-validation 構成

### クラス構成

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

BarkCNN — 軽量 1D CNN（パラメータ数 < 100K）

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
├── __init__.py      # パッケージ初期化
├── __main__.py      # エントリポイント
├── config.py        # ハイパーパラメータ設定
├── dataset.py       # データセット準備・Dataset クラス
├── model.py         # BarkCNN モデル定義
├── train.py         # 学習ループ + ONNX エクスポート
└── README.md        # このファイル
```
