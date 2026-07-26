# プロジェクト構成・環境規約

## Python バージョン

Python 3.12 を使用する。

## 仮想環境

venv を使用する。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## パッケージ管理

`pyproject.toml` で管理する。`pip install -e ".[dev]"` で開発用依存も含めてインストールできるようにする。

## ディレクトリ構成

src レイアウトを採用する。

```
bark-check/
├── src/
│   └── bark_check/
│       ├── __init__.py
│       ├── bark_detector.py     # BarkDetector コアロジック
│       ├── feature_extractor.py # FeatureExtractor
│       ├── models.py            # DetectionResult データモデル
│       ├── audio_loader.py      # AudioLoader（CLI レイヤー）
│       ├── output_formatter.py  # OutputFormatter（CLI レイヤー）
│       ├── coreml_exporter.py   # CoreML エクスポート
│       └── main.py              # CLI エントリポイント
├── tests/
│   ├── test_bark_detector.py
│   ├── test_detection_result.py
│   ├── test_feature_extractor.py
│   ├── test_audio_loader.py
│   ├── test_cli.py
│   └── integration/
│       └── test_end_to_end.py
├── models/                      # 学習済みモデルファイル置き場
├── pyproject.toml
└── README.md
```

## 主要ライブラリ

| 用途 | ライブラリ |
|---|---|
| 音声ファイル読み込み | `soundfile` |
| 音声処理・MFCC 抽出 | `librosa` |
| 推論エンジン | `onnxruntime` |
| CoreML エクスポート | `coremltools >= 7.0` |
| テスト | `pytest` |
| プロパティベーステスト | `hypothesis` |

## CLI エントリポイント

`pyproject.toml` の `[project.scripts]` で `bark-check` コマンドを定義する。

```toml
[project.scripts]
bark-check = "bark_check.main:main"
```
