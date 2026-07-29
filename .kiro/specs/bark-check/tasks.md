# Implementation Plan: bark-check

## Overview

bark-check の完全な実装計画。CLI 推論パイプラインと学習パイプラインの両方を含む。
データモデル → コアロジック → CLI レイヤー → 学習基盤 → モデル実装 → エクスポート・検証の順に進める。

---

## Tasks

- [x] 1. プロジェクト基盤のセットアップ
  - `pyproject.toml` を作成し、依存関係（soundfile, librosa, onnxruntime, pytest, hypothesis, torch, onnx）と CLI エントリポイントを定義する
  - `src/bark_check/__init__.py` を作成してパッケージとして認識させる
  - `tests/__init__.py`、`tests/integration/__init__.py` を作成する
  - `models/` ディレクトリを作成し、`.gitkeep` を配置する
  - _Requirements: 1.1, 4.1_

- [x] 2. DetectionResult データモデルの実装
  - [x] 2.1 `src/bark_check/models.py` に `DetectionResult` dataclass を実装する
    - フィールド: `is_bark`, `confidence`, `timestamp`, `audio_duration`, `error`
    - `to_json()` メソッドと `from_json()` クラスメソッドを実装する
    - _Requirements: 2.1, 5.4, 5.5, 6.1, 6.2_
  - [x] 2.2 Property 8 のプロパティテスト（ラウンドトリップ）
    - **Validates: Requirements 6.1, 6.2, 6.3**
  - [x] 2.3 Property 9 のプロパティテスト（不正 JSON）
    - **Validates: Requirements 6.4**
  - [x] 2.4 Property 10 のプロパティテスト（必須フィールド欠落）
    - **Validates: Requirements 6.5**
  - [x] 2.5 ユニットテストを実装する
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 3. チェックポイント — データモデルのテストがすべて通ること

- [x] 4. FeatureExtractor の実装
  - [x] 4.1 `src/bark_check/feature_extractor.py` に `FeatureExtractor` クラスを実装する
    - `extract(pcm, sample_rate, fixed_length=None)` メソッド
    - 16kHz リサンプリング、MFCC 40 次元、フレームサイズ 400、フレームシフト 160
    - 可変長モード: 出力 [T, 40]、固定長モード: 出力 [fixed_length, 40]
    - _Requirements: 2.6, 4.3, 12.1, 12.2_
  - [x] 4.2 ユニットテスト + Property 13（固定長モード PBT）を実装する
    - _Requirements: 2.6, 4.3_

- [x] 5. BarkDetector コアロジックの実装
  - [x] 5.1 `src/bark_check/bark_detector.py` に `BarkDetector` クラスを実装する
    - モデル自動判別ロジック（4D/3D channels-first/3D channels-last）
    - 空入力・上限超過・無音・推論エラーのハンドリング
    - _Requirements: 2.1-2.7, 4.1, 4.2, 5.1-5.6, 12.1-12.5_
  - [x] 5.2 Property 1（有効 PCM → 有効 DetectionResult）
    - **Validates: Requirements 2.1, 5.4**
  - [x] 5.3 Property 2（閾値一貫性）
    - **Validates: Requirements 2.2, 2.3**
  - [x] 5.4 Property 3（時間長超過エラー）
    - **Validates: Requirements 2.7**
  - [x] 5.5 Property 6（無音 → confidence 0.0）
    - **Validates: Requirements 5.3**
  - [x] 5.6 Property 7（推論エラー格納）
    - **Validates: Requirements 5.1, 5.6**
  - [x] 5.7 ユニットテスト
    - _Requirements: 2.4, 2.5, 5.1, 5.3_

- [x] 6. チェックポイント — コアロジックのテストがすべて通ること

- [x] 7. AudioLoader の実装
  - [x] 7.1 `src/bark_check/audio_loader.py` に `AudioLoader` クラスを実装する
    - FileNotFoundError, UnsupportedFormatError, AudioLoadError
    - _Requirements: 1.2, 1.3, 1.6_
  - [x] 7.2 ユニットテストを実装する
    - _Requirements: 1.2, 1.3, 1.6_

- [x] 8. OutputFormatter の実装
  - [x] 8.1 `src/bark_check/output_formatter.py` に `OutputFormatter` クラスを実装する
    - `format_text()`, `format_json()` メソッド
    - _Requirements: 3.1, 3.2_

- [x] 9. CLI エントリポイントの実装
  - [x] 9.1 `src/bark_check/main.py` に `main()` 関数を実装する
    - argparse で audio_file, --json, --threshold, --model を定義する
    - 各エラー条件に対応する終了コードを返す
    - _Requirements: 1.1-1.6, 3.1-3.7, 5.2_
  - [x] 9.2 Property 4（無効拡張子 → 終了コード 2）
    - **Validates: Requirements 1.3**
  - [x] 9.3 Property 5（--json 出力スキーマ）
    - **Validates: Requirements 3.2**
  - [x] 9.4 CLI ユニットテスト
    - _Requirements: 1.2, 1.4, 1.5, 3.2, 3.4, 3.5, 3.6, 5.2_

- [x] 10. チェックポイント — CLI を含む全テストが通ること

- [x] 11. チェックポイント — 推論パイプライン完成確認

- [x] 12. 最終チェックポイント（推論パイプライン） — 全テストが通ること

- [x] 13. 学習インフラストラクチャの実装
  - [x] 13.1 `training/config.py` に `TrainingConfig` データクラスを実装する
    - data_dir, output_model_path, positive/negative_classes, val_fold
    - sample_rate, clip_duration_sec, epochs, batch_size, learning_rate
    - fixed_frame_length=199, dropout_rate=0.3, use_augmentation=True
    - model_type="conv2d", augmentation_probability=0.5, random_seed=42
    - _Requirements: 7.2, 7.7_
  - [x] 13.2 `training/dataset.py` に ESC50BarkDataset と download_esc50() を実装する
    - ESC-50 自動ダウンロード・展開
    - メタデータ CSV 読み込み、正例/負例フィルタリング
    - ランダムクロップ（学習）/ 中央クロップ（バリデーション）
    - クラスバランス調整（正例オーバーサンプリング）
    - model_type に応じた特徴量出力（conv2d: [1,40,199], conv1d: [T,40]）
    - _Requirements: 7.1, 7.3, 7.4, 10.1-10.5_
  - [x] 13.3 `training/augmentation.py` にデータ拡張関数を実装する
    - `apply_time_shift(pcm, max_shift=1600)`: ±1600 サンプルのランダムシフト
    - `apply_gaussian_noise(pcm, snr_min=20.0, snr_max=40.0)`: ガウシアンノイズ付加
    - _Requirements: 10.1, 10.2, 10.5_
  - [x] 13.4 Property 14（データ拡張の信号長・有限性保存）のテスト
    - **Validates: Requirements 10.1, 10.2, 10.5**
  - [x] 13.5 Property 15（バリデーションモードの決定論性）のテスト
    - **Validates: Requirements 10.3**
  - [x] 13.6 TrainingConfig ユニットテスト
    - _Requirements: 7.2, 7.7_

- [x] 14. Conv1d モデル（BarkCNN）の実装
  - [x] 14.1 `training/model.py` に `BarkCNN` クラスを実装する
    - 入力 [B, T, 40] → permute → Conv1d ブロック ×3 → GAP → Linear → Sigmoid
    - _Requirements: 9.1-9.5_
  - [x] 14.2 `training/train.py` に学習ループ（train_one_epoch, evaluate）を実装する
    - 可変長 collate 関数（ゼロパディング）
    - _Requirements: 7.4_
  - [x] 14.3 ONNX エクスポート（動的軸あり、conv1d 用）
    - _Requirements: 7.5, 9.5_

- [x] 15. Conv2d モデル（BarkCNN2d）の実装
  - [x] 15.1 `training/model.py` に `BarkCNN2d` クラスを実装する
    - 入力 [B, 1, 40, 199] → permute [B, 40, 1, 199] → Conv2d ブロック ×3
    - → AdaptiveAvgPool2d → Dropout → Linear → Sigmoid
    - パラメータ数 82,497（200,000 以下）
    - _Requirements: 8.1-8.7_
  - [x] 15.2 学習パイプラインへの統合（model_type="conv2d" 分岐）
    - 固定長 collate 関数
    - _Requirements: 7.3_
  - [x] 15.3 Property 16（Conv2d/Conv1d 数値的等価性）のテスト
    - **Validates: Requirements 8.1-8.4**
  - [x] 15.4 Property 11（ONNX 推論出力範囲）のテスト
    - **Validates: Requirements 7.6, 8.5**
  - [x] 15.5 Property 12（データセット出力形状一貫性）のテスト
    - **Validates: Requirements 8.1, 10.3**

- [x] 16. ONNX エクスポートと CoreML 互換性検証
  - [x] 16.1 `training/onnx_validator.py` に `validate_onnx_for_coreml()` を実装する
    - 入力 shape [1,1,40,199], 出力 [1,1], opset 9, 全軸固定, 許可オペレータ検証
    - _Requirements: 11.1-11.7_
  - [x] 16.2 `training/train.py` の `export_onnx()` に conv2d 対応を追加する
    - 4D 静的入力 [1,1,40,199] でエクスポート、opset_version=9
    - _Requirements: 7.5, 11.1_
  - [x] 16.3 `training/train.py` の `verify_onnx()` にサニティチェックを実装する
    - onnxruntime で推論し出力が 0.0〜1.0 の範囲内であることを確認
    - _Requirements: 7.6_
  - [x] 16.4 ONNX Validator ユニットテスト
    - _Requirements: 11.1-11.5_

- [x] 17. BarkDetector モデル自動判別の実装
  - [x] 17.1 `src/bark_check/bark_detector.py` の `_load_model()` に 4D 判別を追加する
    - [1,1,40,N] → _is_4d=True, _channels_first=True, _fixed_length=N
    - [1,40,N] → _channels_first=True, _fixed_length=N
    - [1,T,40] → _channels_first=False, _fixed_length=None
    - _Requirements: 12.1-12.5_
  - [x] 17.2 `detect()` の推論パスに 4D 入力対応を追加する
    - channels-first + unsqueeze で [1,1,40,N] を構築
    - _Requirements: 12.1, 12.5_
  - [x] 17.3 モデル自動判別ユニットテスト
    - _Requirements: 12.1-12.4_

- [x] 18. FeatureExtractor 固定長モードの実装
  - [x] 18.1 `extract()` に `fixed_length` パラメータを追加する
    - 末尾ゼロパディング（フレーム数 < fixed_length）
    - 先頭切り出し（フレーム数 > fixed_length）
    - _Requirements: 12.1, 12.2_
  - [x] 18.2 Property 13（任意 PCM 長に対する固定長出力）のテスト
    - **Validates: Requirements 12.1, 12.2**

- [x] 19. 最終チェックポイント — 全テスト（推論 + 学習）が通ること

---

## Notes

- 各タスクは対応する要件番号を参照しているため、トレーサビリティが確保されている
- チェックポイントでインクリメンタルな動作確認を行う
- プロパティテストは Hypothesis の `@settings(max_examples=100)` を使用する
- ONNX モデルが `models/` ディレクトリに存在しない場合、推論テストはモックを使用する
- 学習パイプラインの End-to-End テストは `--epochs 2` の短縮実行で ONNX 検証まで通ることを確認する

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
    { "id": 6, "tasks": ["9.2", "9.3", "9.4"] },
    { "id": 7, "tasks": ["13.1", "13.3"] },
    { "id": 8, "tasks": ["13.2", "13.4", "13.5", "13.6"] },
    { "id": 9, "tasks": ["14.1", "14.2", "15.1"] },
    { "id": 10, "tasks": ["14.3", "15.2", "15.3", "15.4", "15.5"] },
    { "id": 11, "tasks": ["16.1", "16.2", "16.3", "16.4"] },
    { "id": 12, "tasks": ["17.1", "17.2", "17.3", "18.1", "18.2"] }
  ]
}
```
