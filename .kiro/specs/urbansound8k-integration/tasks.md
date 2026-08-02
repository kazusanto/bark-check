# Implementation Plan: UrbanSound8K Integration

## Overview

UrbanSound8K データセットを既存の犬の吠え声検出学習パイプラインに統合する。抽象基底クラス `BarkDatasetBase` を導入し、ESC-50 と UrbanSound8K を統一的に扱える `DatasetFactory` を構築する。既存の ESC-50 のみ学習パイプラインの後方互換性を維持しつつ、約 1,000 サンプルの dog_bark データを追加可能にする。

## Tasks

- [x] 1. TrainingConfig へのデータソース設定追加
  - [x] 1.1 TrainingConfig に data_sources, urbansound8k_dir, urbansound8k_positive_classes, urbansound8k_negative_classes, urbansound8k_val_fold フィールドを追加する
    - `data_sources: list[str]` デフォルト `["esc50"]`
    - `urbansound8k_dir: Path` デフォルト `Path("data/UrbanSound8K")`
    - `urbansound8k_positive_classes: list[int]` デフォルト `[3]`
    - `urbansound8k_negative_classes: list[int]` デフォルト `[2, 8, 1, 5]`
    - `urbansound8k_val_fold: int` デフォルト `10`
    - _Requirements: 4.1, 4.2, 4.3, 2.2, 2.3_

- [x] 2. BarkDatasetBase 抽象基底クラスの実装
  - [x] 2.1 `training/dataset_base.py` に BarkDatasetBase を実装する
    - `torch.utils.data.Dataset` と `ABC` を継承
    - 抽象メソッド `load_entries()` と `_get_val_fold()` を定義
    - `__getitem__` で共通処理（音声読み込み、クロップ/ゼロパディング、データ拡張、MFCC 抽出）を実装
    - `FileNotFoundError` を `__getitem__` で raise（ファイル不在時）
    - conv2d: shape [1, 40, fixed_frame_length]、conv1d: shape [T, 40] のテンソルを返す
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 1.2_

- [x] 3. ESC50BarkDataset のリファクタリング
  - [x] 3.1 `training/dataset.py` の ESC50BarkDataset を BarkDatasetBase 継承に書き換える
    - コンストラクタ引数 `(config: TrainingConfig, *, is_train: bool)` を維持
    - `load_entries()` で ESC-50 メタデータ CSV を読み込み、`{"filepath": Path, "label": int, "fold": int}` 形式で返す
    - `_get_val_fold()` で `config.val_fold` を返す
    - `_balance_classes` ロジックは DatasetFactory に移動するため削除
    - _Requirements: 3.7, 6.3_

  - [ ]* 3.2 既存テストスイートが変更なしでパスすることを確認する
    - ESC50BarkDataset のインタフェース互換性を検証
    - _Requirements: 6.4_

- [x] 4. Checkpoint - 基底クラスとリファクタリングの動作確認
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. UrbanSound8KBarkDataset の実装
  - [x] 5.1 `training/dataset_urbansound8k.py` に UrbanSound8KBarkDataset を実装する
    - BarkDatasetBase を継承
    - `load_entries()` で UrbanSound8K メタデータ CSV から正例・負例エントリを読み込む
    - classID が `urbansound8k_positive_classes` に含まれる → label=1
    - classID が `urbansound8k_negative_classes` に含まれる → label=0
    - ファイルパスを `urbansound8k_dir / audio / fold{fold} / {slice_file_name}` として解決
    - `_get_val_fold()` で `config.urbansound8k_val_fold` を返す
    - _Requirements: 1.1, 1.3, 2.1_

- [x] 6. DatasetFactory の実装
  - [x] 6.1 `training/dataset_factory.py` に `build_dataset()`, `_validate_config()`, `_check_urbansound8k_available()`, `_apply_class_balance()` を実装する
    - `_validate_config()`: 未知ソース名チェック、classID 範囲チェック（0〜9）、positive/negative 重複チェック
    - `_check_urbansound8k_available()`: メタデータ CSV 存在確認、FileNotFoundError にダウンロード URL と期待ディレクトリ構成を含める
    - `build_dataset()`: data_sources ごとに Dataset インスタンス化、複数ソースは ConcatDataset で結合
    - `_apply_class_balance()`: is_train=True のとき正例をオーバーサンプリング（repeat_factor + remainder 方式）、is_train=False のときバランス調整なし
    - 単一データソースのときは ConcatDataset を使わず直接返す
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 2.4, 2.5, 4.5, 7.1, 7.2, 7.3_

  - [ ]* 6.2 DatasetFactory のユニットテストを `tests/test_dataset_factory.py` に実装する
    - `_validate_config` のエラーケース（未知ソース、classID 範囲外、重複）
    - `_check_urbansound8k_available` の FileNotFoundError メッセージ内容
    - `build_dataset` の単一/複数データソース結合ロジック
    - クラスバランス調整の on/off（is_train=True/False）
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1_

- [x] 7. train.py の CLI 引数追加と build_dataset() 統合
  - [x] 7.1 train.py に `--data-sources` CLI 引数を追加し、データセット構築を `build_dataset()` に委譲する
    - `--data-sources` はカンマ区切り文字列、各要素の前後空白を除去
    - 省略時は `data_sources` デフォルト値 `["esc50"]` を使用
    - `ESC50BarkDataset` の直接インスタンス化を `build_dataset(config, is_train=...)` に置き換え
    - 無効なデータソース名は有効ソース一覧を表示してエラー終了
    - _Requirements: 4.4, 4.5, 6.1, 6.2_

- [x] 8. Checkpoint - 全体統合の動作確認
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. プロパティベーステストの実装
  - [ ]* 9.1 Property 1: ラベル割り当て正確性のテスト
    - **Property 1: ラベル割り当て正確性**
    - Hypothesis でメタデータエントリとクラス設定を生成し、label 値が classID/target とクラスリストの関係に基づき正しいことを検証
    - **Validates: Requirements 1.1, 2.1**

  - [ ]* 9.2 Property 2: ゼロパディング不変量のテスト
    - **Property 2: ゼロパディング不変量**
    - clip_length_samples 未満の音声配列を生成し、パディング後の長さが clip_length_samples と等しいことを検証
    - **Validates: Requirements 1.2**

  - [ ]* 9.3 Property 3: Fold 分割正確性のテスト
    - **Property 3: Fold 分割正確性**
    - エントリリストと val_fold を生成し、is_train=True では val_fold エントリが含まれず、is_train=False では val_fold エントリのみ含まれることを検証
    - **Validates: Requirements 1.3, 5.4**

  - [ ]* 9.4 Property 4: 設定バリデーションのテスト
    - **Property 4: 設定バリデーション**
    - 不正な data_sources、重複 classID、範囲外 classID を生成し、ValueError が発生することを検証
    - **Validates: Requirements 2.4, 2.5, 4.5, 5.6**

  - [ ]* 9.5 Property 5: エントリ契約のテスト
    - **Property 5: エントリ契約**
    - load_entries() の返却値が "filepath"（Path）、"label"（0 or 1）、"fold"（≥1）のキーを含むことを検証
    - **Validates: Requirements 3.3**

  - [ ]* 9.6 Property 6: 出力テンソル形状のテスト
    - **Property 6: 出力テンソル形状**
    - model_type ごとに __getitem__ の出力テンソル shape と dtype を検証
    - **Validates: Requirements 3.4**

  - [ ]* 9.7 Property 7: クラスバランスのテスト
    - **Property 7: クラスバランス**
    - 正例・負例を含むエントリリストを生成し、オーバーサンプリング後の |positives - negatives| ≤ 1 を検証
    - **Validates: Requirements 5.3**

  - [ ]* 9.8 Property 8: ConcatDataset 長さの加法性のテスト
    - **Property 8: ConcatDataset 長さの加法性**
    - 複数 Dataset を結合し、結合後の長さが個別長の合計に等しいことを検証（バランス調整前）
    - **Validates: Requirements 5.2**

- [x] 10. README の更新
  - [x] 10.1 `training/README.md` に UrbanSound8K データソースの使い方、CLI 引数 `--data-sources`、手動ダウンロード手順を追記する
    - データソース指定の使用例
    - UrbanSound8K のディレクトリ配置構成
    - ライセンス情報（CC BY-NC 4.0）
    - _Requirements: 7.1_

- [x] 11. Final checkpoint - 全テストパスと後方互換性の最終確認
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- タスク `*` マーク付きはオプションでスキップ可能
- 各タスクは対応する Requirements を参照しトレーサビリティを確保
- チェックポイントでインクリメンタルに動作確認
- プロパティベーステストは設計書の Correctness Properties セクションに基づく
- ユニットテストは特定の例とエッジケースを検証する

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["3.1"] },
    { "id": 3, "tasks": ["3.2", "5.1"] },
    { "id": 4, "tasks": ["6.1"] },
    { "id": 5, "tasks": ["6.2", "7.1"] },
    { "id": 6, "tasks": ["9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8"] },
    { "id": 7, "tasks": ["10.1"] }
  ]
}
```
