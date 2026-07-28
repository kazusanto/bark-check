# Implementation Plan: bark-check

## Overview

bark-check の実装を、データモデル → コアロジック → CLI レイヤーの順に進める。
各ステップは前のステップの成果物を前提とし、テストを通じてインクリメンタルに動作を確認する。
設計書（design.md）の Correctness Properties に対応するプロパティベーステストを各実装タスクに付属させる。

---

## Tasks

- [x] 1. プロジェクト基盤のセットアップ
  - `pyproject.toml` を作成し、依存関係（soundfile, librosa, onnxruntime, pytest, hypothesis）と CLI エントリポイント（`bark-check = "bark_check.main:main"`）を定義する
  - `src/bark_check/__init__.py` を作成してパッケージとして認識させる
  - `tests/__init__.py`、`tests/integration/__init__.py` を作成する
  - `models/` ディレクトリを作成し、`.gitkeep` を配置する
  - _Requirements: 1.1, 4.1_

- [x] 2. DetectionResult データモデルの実装
  - [x] 2.1 `src/bark_check/models.py` に `DetectionResult` dataclass を実装する
    - フィールド: `is_bark: bool`, `confidence: float`, `timestamp: float`, `audio_duration: float`, `error: str | None = None`
    - `to_json()` メソッド（is_bark, confidence, timestamp, audio_duration, error を JSON にシリアライズ）を実装する
    - `from_json()` クラスメソッド（不正 JSON / 必須フィールド欠落時は error フィールド付き DetectionResult を返し例外を発生させない）を実装する
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 2.1, 5.4, 5.5, 6.1, 6.2_

  - [x] 2.2 Property 8 のプロパティテストを `tests/test_detection_result.py` に実装する
    - **Property 8: DetectionResult のシリアライズ・デシリアライズ ラウンドトリップ**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 2.3 Property 9 のプロパティテストを `tests/test_detection_result.py` に実装する
    - **Property 9: 不正 JSON は error フィールドを持つ DetectionResult を返す**
    - **Validates: Requirements 6.4**

  - [x] 2.4 Property 10 のプロパティテストを `tests/test_detection_result.py` に実装する
    - **Property 10: 必須フィールド欠落 JSON は欠落フィールド名をエラーに含む DetectionResult を返す**
    - **Validates: Requirements 6.5**

  - [x] 2.5 `tests/test_detection_result.py` にユニットテストを実装する
    - 正常シリアライズ・デシリアライズのケース（error=None, error あり）をテストする
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 3. チェックポイント — データモデルのテストがすべて通ること
  - すべてのテストが通ることを確認する。問題があればユーザーに質問する。

- [x] 4. FeatureExtractor の実装
  - [x] 4.1 `src/bark_check/feature_extractor.py` に `FeatureExtractor` クラスを実装する
    - `extract(pcm: np.ndarray, sample_rate: int) -> np.ndarray` メソッドを実装する
    - librosa を使って MFCC を抽出する（サンプリングレート 16,000 Hz にリサンプリング、フレームサイズ 400、フレームシフト 160、MFCC 係数数 40）
    - 出力形状: `[T, 40]`
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 2.6, 4.3_

  - [x] 4.2 `tests/test_feature_extractor.py` にユニットテストを実装する
    - 16kHz の正弦波 PCM を入力したとき出力形状が `[T, 40]` になることをテストする
    - 8kHz の入力を渡したとき 16kHz へのリサンプリングが正しく動作することをテストする
    - _Requirements: 2.6, 4.3_

- [x] 5. BarkDetector コアロジックの実装
  - [x] 5.1 `src/bark_check/bark_detector.py` に `BarkDetector` クラスと `ModelLoadError` 例外クラスを実装する
    - `__init__(self, threshold: float = 0.5, model_path: str | None = None)` を実装する（threshold が [0.0, 1.0] 範囲外なら ValueError）
    - `detect(self, pcm: np.ndarray, sample_rate: int) -> DetectionResult` を実装する
      - 空入力（length == 0）: `error="Input PCM block is empty"` を持つ DetectionResult を返す
      - 上限超過（length > 32000）: `error="Input exceeds maximum length of 32000 samples"` を持つ DetectionResult を返す
      - 無音入力（全ゼロ）: `is_bark=False`, `confidence=0.0`, `error=None` を返す
      - 推論中ランタイムエラー: `error="Inference error: <message>"` を持つ DetectionResult を返し例外を伝播しない
      - 正常時: ONNX Runtime で推論し confidence を計算、threshold と比較して is_bark を決定する
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.8, 4.1, 4.2, 5.1, 5.3, 5.4, 5.5, 5.6_

  - [x] 5.2 Property 1 のプロパティテストを `tests/test_bark_detector.py` に実装する
    - **Property 1: 有効な PCM 入力は常に有効な DetectionResult を返す**
    - **Validates: Requirements 2.1, 5.4**

  - [x] 5.3 Property 2 のプロパティテストを `tests/test_bark_detector.py` に実装する
    - **Property 2: 閾値による吠え声判定の一貫性**
    - **Validates: Requirements 2.2, 2.3**

  - [x] 5.4 Property 3 のプロパティテストを `tests/test_bark_detector.py` に実装する
    - **Property 3: 上限超過入力はエラー DetectionResult を返す**
    - **Validates: Requirements 2.8**

  - [x] 5.5 Property 6 のプロパティテストを `tests/test_bark_detector.py` に実装する
    - **Property 6: 無音入力は常に confidence 0.0 の吠え声なしを返す**
    - **Validates: Requirements 5.3**

  - [x] 5.6 Property 7 のプロパティテストを `tests/test_bark_detector.py` に実装する
    - **Property 7: 推論エラーは DetectionResult に格納され例外は伝播しない**
    - **Validates: Requirements 5.1, 5.6**

  - [x] 5.7 `tests/test_bark_detector.py` にユニットテストを実装する
    - デフォルト閾値が 0.5 であることをテストする（Requirements 2.4）
    - 空入力エラーをテストする（Requirements 5.1）
    - 無音入力で confidence=0.0 を返すことをテストする（Requirements 5.3）
    - 2 秒の PCM で推論が 500ms 以内に完了することをベンチマークとしてテストする（Requirements 2.5）
    - _Requirements: 2.4, 2.5, 5.1, 5.3_

- [x] 6. チェックポイント — コアロジックのテストがすべて通ること
  - すべてのテストが通ることを確認する。問題があればユーザーに質問する。

- [x] 7. AudioLoader の実装
  - [x] 7.1 `src/bark_check/audio_loader.py` に `AudioLoader` クラスと `UnsupportedFormatError`、`AudioLoadError` 例外クラスを実装する
    - `SUPPORTED_FORMATS = ("wav", "mp3", "flac", "ogg")`
    - `load(self, file_path: str) -> tuple[np.ndarray, int]` を実装する
      - ファイル非存在: `FileNotFoundError` を送出する
      - 非対応拡張子: `UnsupportedFormatError` を送出する
      - デコード失敗: `AudioLoadError` を送出する
      - 正常時: soundfile / librosa でデコードしてモノラル float32 PCM に変換して返す
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 1.2, 1.3, 1.6_

  - [x] 7.2 `tests/test_audio_loader.py` にユニットテストを実装する
    - 存在しないパスで FileNotFoundError が送出されることをテストする
    - 非対応拡張子で UnsupportedFormatError が送出されることをテストする
    - 正常な WAV バイトで (pcm, sample_rate) タプルが返ることをテストする（numpy を使って合成 PCM を WAV バッファに書き込んでテスト）
    - _Requirements: 1.2, 1.3, 1.6_

- [x] 8. OutputFormatter の実装
  - [x] 8.1 `src/bark_check/output_formatter.py` に `OutputFormatter` クラスを実装する
    - `format_text(self, result: DetectionResult) -> str` を実装する（吠え声あり／なし + confidence を小数点以下 2 桁で返す）
    - `format_json(self, result: DetectionResult) -> str` を実装する（DetectionResult.to_json() を利用）
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 3.1, 3.2_

- [x] 9. CLI エントリポイントの実装
  - [x] 9.1 `src/bark_check/main.py` に `main()` 関数を実装する
    - argparse で `audio_file`（位置引数）、`--json`、`--threshold` オプションを定義する
    - `--threshold` が [0.0, 1.0] 範囲外のとき stderr にエラーを出力して終了コード 1 で終了する
    - AudioLoader, BarkDetector, OutputFormatter を組み合わせてエンドツーエンドの処理フローを実装する
    - 各エラー条件に対応する終了コード（0, 1, 2, 3, 4）を返す
    - docstring は日本語 Google スタイルで記述する
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 5.2_

  - [x] 9.2 Property 4 のプロパティテストを `tests/test_cli.py` に実装する
    - **Property 4: 無効拡張子は常に終了コード 2 で拒否される**
    - **Validates: Requirements 1.3**

  - [x] 9.3 Property 5 のプロパティテストを `tests/test_cli.py` に実装する
    - **Property 5: --json 出力は常に有効なスキーマを持つ**
    - **Validates: Requirements 3.2**

  - [x] 9.4 `tests/test_cli.py` にユニットテストを実装する
    - `--help` で使用方法と対応フォーマット一覧が表示されることをテストする（Requirements 1.4）
    - 引数未指定で終了コード 1 になることをテストする（Requirements 1.5）
    - 存在しないパスで終了コード 1 になることをテストする（Requirements 1.2）
    - 吠え声あり判定で終了コード 0 になることをテストする（Requirements 3.4）
    - 吠え声なし判定で終了コード 3 になることをテストする（Requirements 3.5）
    - `--json` で JSON が標準出力に出力されることをテストする（Requirements 3.2）
    - `--threshold 0.7` が BarkDetector に渡されることをテストする（Requirements 3.6）
    - モデル読み込み失敗で終了コード 4 になることをテストする（Requirements 5.2）
    - _Requirements: 1.2, 1.4, 1.5, 3.2, 3.4, 3.5, 3.6, 5.2_

- [x] 10. チェックポイント — CLI を含む全テストが通ること
  - すべてのテストが通ることを確認する。問題があればユーザーに質問する。

- [x] 12. 最終チェックポイント — 全テストが通ること
  - すべてのテストが通ることを確認する。問題があればユーザーに質問する。

---

## Notes

- `*` が付いたサブタスクはオプション（MVP では省略可能）
- 各タスクは対応する要件番号を参照しているため、トレーサビリティが確保されている
- チェックポイントでインクリメンタルな動作確認を行う
- プロパティテストは Hypothesis の `@settings(max_examples=100)` を使用し、各テストに `# Feature: bark-check, Property N: <タイトル>` 形式のコメントを付与する
- ユニットテストは `pytest` を使用する
- ONNX モデルが `models/` ディレクトリに存在しない場合、推論テストはモックを使用する

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "4.1"] },
    { "id": 3, "tasks": ["4.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "7.1", "8.1"] },
    { "id": 5, "tasks": ["7.2", "9.1"] },
    { "id": 6, "tasks": ["9.2", "9.3", "9.4"] }
  ]
}
```
