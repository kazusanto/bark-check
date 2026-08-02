# Requirements Document

## Introduction

UrbanSound8K データセットを統合し、犬の吠え声検出モデルの学習データを大幅に拡充する。ESC-50 の dog_bark クラスは 40 サンプルのみであるのに対し、UrbanSound8K は約 1,000 サンプルの dog_bark を含む。Dataset 層を抽象化・汎用化し、複数データソースを統一的に扱えるアーキテクチャに再設計する。既存の ESC-50 のみでの学習パイプラインは後方互換性を維持する。

## Glossary

- **BarkDatasetBase**: 複数データソースに対応する抽象基底クラス。共通処理（ランダムクロップ、MFCC 抽出、データ拡張）を集約する。
- **ESC50BarkDataset**: ESC-50 データソースに特化した BarkDatasetBase のサブクラス。
- **UrbanSound8KBarkDataset**: UrbanSound8K データソースに特化した BarkDatasetBase のサブクラス。
- **TrainingConfig**: 学習パイプラインのハイパーパラメータとパス設定を管理する dataclass。
- **DatasetFactory**: TrainingConfig に基づいて適切な Dataset インスタンスを構築するファクトリ関数群。
- **ConcatDataset**: PyTorch 標準の ConcatDataset。複数 Dataset を結合して単一の Dataset として扱う。
- **UrbanSound8K**: 都市環境音を 10 クラスに分類した音響データセット。10-fold 構成、各クリップは 4 秒以下の WAV ファイル。
- **ESC-50**: 環境音を 50 クラスに分類した音響データセット。5-fold 構成、各クリップは 5 秒の WAV ファイル。
- **classID**: UrbanSound8K におけるクラス識別子（整数値）。
- **dog_bark**: 犬の吠え声クラス。ESC-50 では target=0、UrbanSound8K では classID=3。
- **data_sources**: TrainingConfig のフィールド。使用するデータソースを文字列リストで指定する（例: `["esc50"]`, `["esc50", "urbansound8k"]`）。
- **load_entries**: BarkDatasetBase のサブクラスが実装する抽象メソッド。データソース固有のメタデータを読み込み、統一形式のエントリリストを返す。

## Requirements

### Requirement 1: UrbanSound8K dog_bark の正例利用

**User Story:** As a ML エンジニア, I want UrbanSound8K の dog_bark クラス（約 1,000 サンプル）を正例として学習に利用したい, so that 犬の吠え声検出モデルの汎化性能を向上させられる。

#### Acceptance Criteria

1. WHEN data_sources に "urbansound8k" が含まれる場合, THE UrbanSound8KBarkDataset SHALL UrbanSound8K メタデータ CSV（UrbanSound8K.csv）から classID=3 のエントリを正例（label=1）として読み込み、各エントリの音声ファイルパスを `urbansound8k_dir / audio / fold{fold} / {slice_file_name}` として解決する。
2. WHEN 読み込んだ音声のサンプル数が clip_duration_sec × sample_rate 未満の場合, THE BarkDatasetBase SHALL 音声末尾にゼロパディングを付加して clip_duration_sec × sample_rate サンプルの長さに揃える。
3. THE UrbanSound8KBarkDataset SHALL urbansound8k_val_fold（デフォルト: 10）で指定された fold 番号のエントリをバリデーション用、それ以外の fold（1〜10 のうち val_fold を除く）を学習用として分割する。

### Requirement 2: UrbanSound8K 暮らし系サウンドの負例利用

**User Story:** As a ML エンジニア, I want UrbanSound8K の暮らし系サウンド（children_playing, siren, car_horn, engine_idling）を負例として追加したい, so that 都市環境ノイズに対する偽陽性を低減できる。

#### Acceptance Criteria

1. WHEN data_sources に "urbansound8k" が含まれる場合, THE UrbanSound8KBarkDataset SHALL UrbanSound8K メタデータ CSV から urbansound8k_negative_classes で指定された classID のエントリを負例（label=0）として読み込む。
2. THE TrainingConfig SHALL urbansound8k_negative_classes フィールドを `list[int]` 型で定義し、デフォルト値として [2, 8, 1, 5]（children_playing, siren, car_horn, engine_idling）を保持する。
3. THE TrainingConfig SHALL urbansound8k_positive_classes フィールドを `list[int]` 型で定義し、デフォルト値として [3]（dog_bark）を保持する。
4. IF urbansound8k_positive_classes と urbansound8k_negative_classes に同一の classID が含まれる場合, THEN THE DatasetFactory SHALL 重複する classID を示すエラーメッセージを出力し、データセット構築を中断する。
5. IF urbansound8k_negative_classes に 0〜9 の範囲外の classID が含まれる場合, THEN THE DatasetFactory SHALL 無効な classID 値を示すエラーメッセージを出力し、データセット構築を中断する。

### Requirement 3: Dataset 抽象基底クラスの導入

**User Story:** As a 開発者, I want Dataset クラスを抽象化して複数データソースを統一的に扱いたい, so that 将来のデータソース追加時にも最小限の変更で対応できる。

#### Acceptance Criteria

1. THE BarkDatasetBase SHALL torch.utils.data.Dataset を継承した抽象基底クラスとして、共通処理（ランダムクロップ、MFCC 抽出、データ拡張、ゼロパディング）を `__getitem__` 内に実装する。
2. THE BarkDatasetBase SHALL 抽象メソッド load_entries() を定義し、サブクラスに `list[dict]` 形式のエントリリスト返却を要求する。
3. FOR ALL load_entries() が返すエントリ, THE BarkDatasetBase SHALL 各エントリが "filepath"（音声ファイルの Path）、"label"（0 または 1 の整数）、"fold"（1 以上の正の整数）のキーを含むことを前提とする。
4. WHEN model_type が "conv2d" の場合, THE BarkDatasetBase の `__getitem__` SHALL shape [1, 40, fixed_frame_length] の float32 テンソルとラベル shape [1] のタプルを返す。WHEN model_type が "conv1d" の場合, THE BarkDatasetBase の `__getitem__` SHALL shape [T, 40] の float32 テンソルとラベル shape [1] のタプルを返す。
5. WHILE is_train が True の場合, THE BarkDatasetBase SHALL ランダムクロップとデータ拡張（model_type が "conv2d" の場合のみ）を適用する。WHILE is_train が False の場合, THE BarkDatasetBase SHALL 中央クロップを適用し、データ拡張を適用しない。
6. IF load_entries() が返すエントリの "filepath" が存在しないファイルを指す場合, THEN THE BarkDatasetBase SHALL `__getitem__` 呼び出し時にファイルが見つからないことを示す例外を発生させる。
7. THE ESC50BarkDataset SHALL BarkDatasetBase を継承し、ESC-50 固有のメタデータ読み込みを load_entries() で実装する。
8. THE UrbanSound8KBarkDataset SHALL BarkDatasetBase を継承し、UrbanSound8K 固有のメタデータ読み込みを load_entries() で実装する。

### Requirement 4: TrainingConfig のデータソース指定

**User Story:** As a ML エンジニア, I want TrainingConfig で利用するデータソースを柔軟に指定したい, so that 実験ごとに異なるデータソースの組み合わせで学習を行える。

#### Acceptance Criteria

1. THE TrainingConfig SHALL data_sources フィールド（型: `list[str]`）を持ち、デフォルト値として `["esc50"]` を設定する。
2. THE TrainingConfig SHALL urbansound8k_dir フィールド（型: `Path`）を持ち、デフォルト値として `Path("data/UrbanSound8K")` を設定する。
3. THE TrainingConfig SHALL urbansound8k_val_fold フィールド（型: `int`、有効範囲: 1〜10）を持ち、デフォルト値として 10 を設定する。
4. WHEN CLI 引数 --data-sources が指定された場合, THE 学習スクリプト SHALL カンマ区切りの文字列をパースし、各要素の前後空白を除去した上で data_sources フィールドに設定する（例: "esc50,urbansound8k"）。
5. IF data_sources に "esc50" および "urbansound8k" 以外の文字列が含まれる場合, THEN THE 学習スクリプト SHALL 無効なデータソース名を示すエラーメッセージを出力し、有効なデータソース名の一覧を表示して終了する。
6. IF data_sources に "urbansound8k" が含まれるが urbansound8k_dir 配下にメタデータ CSV（UrbanSound8K/metadata/UrbanSound8K.csv）が存在しない場合, THEN THE DatasetFactory SHALL データセット未ダウンロードを示すエラーメッセージを表示し、手動ダウンロード手順を案内する。

### Requirement 5: Dataset ファクトリと ConcatDataset 結合

**User Story:** As a 開発者, I want ファクトリ関数で設定に基づいた Dataset を自動構築したい, so that データソースの組み合わせロジックを呼び出し側から分離できる。

#### Acceptance Criteria

1. THE DatasetFactory SHALL build_dataset(config, is_train) 関数を提供し、TrainingConfig の data_sources の各要素に対応する Dataset インスタンスを構築する。
2. WHEN data_sources に複数のデータソースが指定された場合, THE DatasetFactory SHALL ConcatDataset を使用して各データソースの Dataset を結合し、結合後の Dataset を返す。
3. WHEN is_train=True かつ結合後（または単一）のデータセットに正例と負例の両方が存在する場合, THE DatasetFactory SHALL 正例を負例の件数と一致するようオーバーサンプリング（repeat_factor + remainder 方式）してクラスバランスを調整する。
4. WHEN is_train=False の場合, THE DatasetFactory SHALL クラスバランス調整を適用せず、データソースから取得したバリデーション用エントリをそのまま返す。
5. WHEN data_sources に単一のデータソースのみが指定された場合, THE DatasetFactory SHALL ConcatDataset を使用せず該当する単一の Dataset インスタンスを返す。
6. IF data_sources に "esc50" および "urbansound8k" 以外の未知のデータソース名が含まれる場合, THEN THE DatasetFactory SHALL 未対応のデータソース名を示すエラーを発生させる。

### Requirement 6: 後方互換性の維持

**User Story:** As a 既存ユーザー, I want ESC-50 のみでの学習が引き続き動作することを保証したい, so that UrbanSound8K を未ダウンロードの環境でも学習パイプラインが利用できる。

#### Acceptance Criteria

1. WHEN data_sources がデフォルト値 `["esc50"]` のまま使用された場合, THE 学習パイプライン SHALL ESC-50 のみを使用して学習を実行し、UrbanSound8K のファイル読み込み・ディレクトリ参照・メタデータ CSV パースのいずれも行わない。
2. WHEN --data-sources CLI 引数が省略された場合, THE 学習スクリプト SHALL data_sources のデフォルト値 `["esc50"]` を使用する。
3. THE ESC50BarkDataset SHALL BarkDatasetBase を継承した後も、コンストラクタ引数として TrainingConfig と is_train（bool）を受け取るインタフェースを維持し、ESC-50 の fold 分割（val_fold: 1〜5）と正例クラス（positive_classes）・負例クラス（negative_classes）の指定に基づくフィルタリング動作を維持する。
4. FOR ALL 既存のテストケース（tests/ 配下の training 関連ユニットテスト）, THE リファクタリング後の ESC50BarkDataset SHALL テストコードを変更せずに合格する。
5. WHEN data_sources が `["esc50"]` の場合, THE 学習パイプライン SHALL リファクタリング前と同一形状の ONNX モデル（入力名 "input"、出力名 "output"、opset_version 9）をエクスポートする。
6. IF UrbanSound8K のディレクトリが存在しない環境で data_sources が `["esc50"]` のまま学習を実行した場合, THEN THE 学習パイプライン SHALL エラーなく学習を完了し、ONNX モデルファイルを出力する。

### Requirement 7: UrbanSound8K 手動ダウンロード案内

**User Story:** As a ML エンジニア, I want UrbanSound8K が未ダウンロード時に案内メッセージを表示してほしい, so that ライセンス上の制約を理解した上で手動でダウンロードできる。

#### Acceptance Criteria

1. IF data_sources に "urbansound8k" が含まれるが urbansound8k_dir/metadata/UrbanSound8K.csv が存在しない場合, THEN THE DatasetFactory SHALL FileNotFoundError を raise し、エラーメッセージにダウンロード先 URL（https://urbansounddataset.weebly.com/urbansound8k.html）および期待するディレクトリ配置構成（metadata/UrbanSound8K.csv と audio/fold1〜fold10）を含める。
2. IF urbansound8k_dir/metadata/UrbanSound8K.csv が存在しない場合に FileNotFoundError が raise された場合, THEN THE DatasetFactory SHALL 他の data_sources（例: "esc50"）にフォールバックせず、処理を中断する。
3. THE DatasetFactory SHALL UrbanSound8K の自動ダウンロードを行わない（Creative Commons Attribution Non-Commercial 4.0 ライセンスの制約により手動ダウンロードのみ対応）。
