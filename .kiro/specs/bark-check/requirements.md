# Requirements Document

## Introduction

bark-check は、音声ファイルを入力として受け取り、その音声が犬の吠え声（bark）を含むかどうかを判定するツールである。
Python CLI コマンドとして提供され、判定ロジックは iOS Swift アプリなど他のプラットフォームへの転用を考慮したポータブルな設計とする。

判定ロジックはコアライブラリ（BarkDetector）として独立させ、CLI はそのラッパーとして実装する。
BarkDetector は音声ファイルを一切扱わず、メモリ上のモノラル PCM サンプル列（float32 配列）とサンプリングレート（Hz）のみを入力として受け付ける。
これにより、1〜2 秒程度の音声ブロックをリアルタイムで判定するユースケース（iOS Swift アプリなど）に対しても、同一のアルゴリズムを転用できる。

CLI レイヤーでは AudioLoader が音声ファイルをデコードしてモノラル PCM に変換し、BarkDetector に渡す。

学習パイプライン（`training/` ディレクトリ）は ESC-50 および UrbanSound8K データセットを使用して犬の吠え声検出モデルを学習し、ONNX 形式でエクスポートする。Conv1d モデル（可変長 CLI 推論向け）と Conv2d モデル（CoreML 互換固定長向け）の 2 種類のアーキテクチャをサポートする。複数データソースの結合は DatasetFactory が担当し、data_sources 設定により柔軟にデータソースを選択できる。

## Glossary

- **BarkDetector**: モノラル PCM サンプル列を入力として受け取り、犬の吠え声を検出するコアロジックモジュール
- **PcmBlock**: モノラル PCM サンプルの配列。float32 型、サンプリングレートは任意だが 16kHz を推奨
- **AudioLoader**: 音声ファイルをデコードしてモノラル PCM に変換する CLI レイヤーのモジュール
- **DetectionResult**: 判定結果を表すデータ構造。吠え声の有無（boolean）、信頼度スコア（0.0〜1.0）、タイムスタンプ、音声長、エラー情報を含む
- **CLI**: `bark-check` コマンドとして提供されるコマンドラインインターフェース
- **ConfidenceScore**: BarkDetector が出力する予測の確からしさを示す 0.0 以上 1.0 以下の浮動小数点数
- **FeatureExtractor**: PcmBlock から機械学習モデルへの入力特徴量（MFCC）を抽出するモジュール
- **BarkCNN**: Conv1d ベースの可変長入力対応モデル。CLI 推論に最適
- **BarkCNN2d**: Conv2d ベースの固定長入力モデル。CoreML (iOS 12) 互換の ONNX エクスポート向け
- **TrainingConfig**: 学習パイプラインのハイパーパラメータとパス設定を管理するデータクラス
- **ESC50BarkDataset**: ESC-50 データセットから犬の吠え声の正例・負例を構築する PyTorch Dataset クラス
- **OnnxValidator**: ONNX モデルの CoreML 互換性を検証するバリデータモジュール
- **CoreML**: Apple の機械学習フレームワーク。iOS 12 以降で利用可能
- **ONNX**: Open Neural Network Exchange。モデルの相互運用フォーマット
- **Global_Average_Pooling**: 時間軸方向の平均を取ることで任意長入力を固定長ベクトルに変換するプーリング手法
- **Data_Augmentation**: 学習データに変換（タイムシフト、ノイズ付加等）を適用して汎化性能を向上させる手法
- **BarkDatasetBase**: 複数データソースに対応する抽象基底クラス。共通処理（ランダムクロップ、MFCC 抽出、データ拡張）を集約する
- **UrbanSound8KBarkDataset**: UrbanSound8K データソースに特化した BarkDatasetBase のサブクラス
- **DatasetFactory**: TrainingConfig に基づいて適切な Dataset インスタンスを構築するファクトリ関数群
- **ConcatDataset**: PyTorch 標準の ConcatDataset。複数 Dataset を結合して単一の Dataset として扱う
- **UrbanSound8K**: 都市環境音を 10 クラスに分類した音響データセット。10-fold 構成、各クリップは 4 秒以下の WAV ファイル
- **classID**: UrbanSound8K におけるクラス識別子（整数値）
- **dog_bark**: 犬の吠え声クラス。ESC-50 では target=0、UrbanSound8K では classID=3
- **data_sources**: TrainingConfig のフィールド。使用するデータソースを文字列リストで指定する（例: `["esc50"]`, `["esc50", "urbansound8k"]`）
- **load_entries**: BarkDatasetBase のサブクラスが実装する抽象メソッド。データソース固有のメタデータを読み込み、統一形式のエントリリストを返す

---

## Requirements

### Requirement 1: 音声ファイルの入力受け付け

**User Story:** As a 開発者, I want CLI コマンドに音声ファイルパスを引数として渡すことで判定を実行したい, so that スクリプトやパイプラインに組み込める

#### Acceptance Criteria

1. THE CLI SHALL 音声ファイルへのパスを位置引数として 1 つ受け付ける
2. WHEN 指定されたパスが存在しないファイルを指す場合, THE CLI SHALL 当該パスを含むエラーメッセージを標準エラー出力に出力し、終了コード 1 で終了する
3. IF 指定されたファイルの拡張子が対応フォーマット（WAV、MP3、FLAC、OGG）のいずれでもない場合, THEN THE CLI SHALL 対応フォーマットの一覧を含むエラーメッセージを標準エラー出力に出力し、終了コード 2 で終了する
4. THE CLI SHALL `--help` オプションで位置引数の説明と対応フォーマット一覧を含む使用方法を表示する
5. WHEN 位置引数が指定されずに実行された場合, THE CLI SHALL 使用方法を示すエラーメッセージを標準エラー出力に出力し、終了コード 1 で終了する
6. IF 音声ファイルの読み込みに失敗した場合, THEN THE AudioLoader SHALL ファイル読み込みエラーを示すエラーメッセージを標準エラー出力に出力し、終了コード 1 で終了する

---

### Requirement 2: 犬の吠え声の判定

**User Story:** As a 開発者, I want PcmBlock を渡すだけで犬の吠え声かどうかを判定したい, so that 手動でラベル付けする手間を省ける

#### Acceptance Criteria

1. WHEN PcmBlock とサンプリングレートが与えられた場合, THE BarkDetector SHALL 吠え声の有無（真偽値）と 0.0 以上 1.0 以下の ConfidenceScore を含む DetectionResult を返す
2. WHEN DetectionResult の ConfidenceScore が設定された閾値以上である場合, THE BarkDetector SHALL 吠え声あり（`True`）と判定する
3. IF DetectionResult の ConfidenceScore が設定された閾値未満である場合, THEN THE BarkDetector SHALL 吠え声なし（`False`）と判定する
4. THE BarkDetector SHALL ConfidenceScore の閾値のデフォルト値を 0.5 とし、0.0 以上 1.0 以下の範囲で設定可能とする
5. THE BarkDetector SHALL 2 秒以下の PcmBlock に対して 500ms 以内に DetectionResult を返す
6. THE FeatureExtractor SHALL 16kHz のモノラル PCM を前提として特徴量を抽出する
7. IF PcmBlock の時間長 (len(pcm) / sample_rate) が 10.0 秒を超える場合, THEN THE BarkDetector SHALL `is_bark=False`, `confidence=0.0`, `error="Input exceeds maximum duration of 10.0 seconds"` を含む DetectionResult を返す

---

### Requirement 3: 判定結果の出力

**User Story:** As a 開発者, I want 判定結果を標準出力で受け取りたい, so that スクリプトで結果を処理できる

#### Acceptance Criteria

1. WHEN 判定が成功した場合, THE CLI SHALL 判定結果（吠え声あり／なし）と小数点以下 2 桁の ConfidenceScore を標準出力に出力する
2. WHEN `--json` オプションが指定された場合, THE CLI SHALL `is_bark` および `confidence` フィールドを含む DetectionResult を JSON 形式で標準出力に出力する
3. IF 判定中にエラーが発生した場合, THEN THE CLI SHALL エラー内容を標準エラー出力に出力し、終了コード 1 で終了する
4. WHEN 判定結果が吠え声ありの場合, THE CLI SHALL 終了コード 0 で終了する
5. WHEN 判定結果が吠え声なしの場合, THE CLI SHALL 終了コード 3 で終了する
6. THE CLI SHALL `--threshold` オプションで判定閾値（0.0 以上 1.0 以下の浮動小数点数）を指定できる
7. IF `--threshold` オプションに 0.0 未満または 1.0 超の値が指定された場合, THEN THE CLI SHALL エラーメッセージを標準エラー出力に出力し、終了コード 1 で終了する

---

### Requirement 4: ポータブルなコアロジック設計

**User Story:** As a iOS Swift アプリなど他プラットフォームの開発者, I want Python で実装された判定ロジックと同等のアルゴリズムを再実装または移植したい, so that 各プラットフォーム内でオフライン判定できる

#### Acceptance Criteria

1. THE BarkDetector SHALL CLI 固有の依存（argparse 等）を含まず、インポート時に副作用（標準出力への書き込み、プロセス終了等）を発生させずに単独でインポート可能なモジュールとして実装される
2. THE BarkDetector SHALL 入力として float32 型のモノラル PCM サンプル配列とサンプリングレート（Hz）を受け取り、`is_bark`（真偽値）と `confidence`（0.0 以上 1.0 以下の浮動小数点数）を最低限含む DetectionResult を返すインターフェースを持つ
3. THE FeatureExtractor SHALL 使用する特徴量の種類、サンプリングレート（Hz）、フレームサイズ（サンプル数）、フレームシフト（サンプル数）を設計ドキュメントに明記する
4. THE BarkDetector SHALL 使用する事前学習済みモデルのアーキテクチャ、入力テンソルの形状と型、出力テンソルの形状と型を設計ドキュメントに明記する

---

### Requirement 5: エラーハンドリング

**User Story:** As a 開発者, I want 判定中に発生したエラーを適切に処理したい, so that 不正な入力や環境問題に対して安全に失敗できる

#### Acceptance Criteria

1. IF PcmBlock が空（サンプル数ゼロ）である場合, THEN THE BarkDetector SHALL 空入力エラーを示すエラー種別を含む DetectionResult を返し、例外を呼び出し元に伝播しない
2. IF モデルの読み込みに失敗した場合, THEN THE CLI SHALL エラーメッセージを標準エラー出力に出力し、終了コード 4 で終了する
3. IF PcmBlock が無音（全サンプルがゼロ）である場合, THEN THE BarkDetector SHALL 信頼度スコア 0.0 を持つ「吠え声なし」の DetectionResult を返す
4. THE BarkDetector SHALL DetectionResult に常に `error` フィールドを含め、エラーが発生していない場合は `null` を格納する
5. IF エラーが発生した場合, THEN THE BarkDetector SHALL DetectionResult の `error` フィールドに人間が読めるエラーメッセージを格納する
6. IF 推論の実行中に実行時エラーが発生した場合, THEN THE BarkDetector SHALL 推論エラーを示すエラー種別を含む DetectionResult を返し、例外を呼び出し元に伝播しない

---

### Requirement 6: DetectionResult のシリアライズ・デシリアライズ（ラウンドトリップ）

**User Story:** As a 開発者, I want DetectionResult を JSON で保存・復元したい, so that 判定結果をログや API レスポンスとして利用できる

#### Acceptance Criteria

1. THE BarkDetector SHALL DetectionResult の `is_bark`、`confidence`、`timestamp`、`audio_duration` フィールドを含む JSON 文字列にシリアライズする機能を提供する
2. THE BarkDetector SHALL JSON 文字列から DetectionResult をデシリアライズし、元の各フィールド値を復元する機能を提供する
3. WHEN 有効な DetectionResult がシリアライズされてからデシリアライズされた場合, THE BarkDetector SHALL `confidence` を小数点以下 6 桁の精度、`timestamp` を秒単位の精度で保持し、元の DetectionResult と等価なオブジェクトを返す（ラウンドトリップ特性）
4. IF 不正な JSON 文字列がデシリアライズに渡された場合, THEN THE BarkDetector SHALL `error` フィールドにパースエラーを示すメッセージを持つ DetectionResult を返し、処理前の状態を変更しない
5. IF 必須フィールド（`is_bark`、`confidence`）を欠く JSON 文字列がデシリアライズに渡された場合, THEN THE BarkDetector SHALL `error` フィールドに欠落フィールド名を含むエラーメッセージを持つ DetectionResult を返す

---

### Requirement 7: 学習パイプライン

**User Story:** As a 開発者, I want ESC-50 データセットで犬の吠え声検出モデルを学習したい, so that 独自の学習済みモデルを生成してアプリに組み込める

#### Acceptance Criteria

1. THE training pipeline SHALL ESC-50 データセットの自動ダウンロード・展開機能を提供し、既にダウンロード済みの場合はスキップする
2. THE TrainingConfig SHALL model_type として "conv1d"（可変長 CLI 推論向け）または "conv2d"（固定長 CoreML 変換向け）を選択可能にする
3. WHEN model_type が "conv2d" の場合, THE training pipeline SHALL BarkCNN2d インスタンスを生成し学習に使用する
4. WHEN model_type が "conv1d" の場合, THE training pipeline SHALL BarkCNN インスタンスを生成し学習に使用する
5. THE training pipeline SHALL 学習完了後に ONNX 形式（opset_version=9）でモデルをエクスポートする
6. THE training pipeline SHALL ONNX エクスポート後に onnxruntime でサニティチェックを実行し、出力が 0.0〜1.0 の範囲内であることを確認する
7. THE TrainingConfig SHALL エポック数、バッチサイズ、学習率、クロップ長、バリデーション fold、出力パスなどのハイパーパラメータを設定可能にする

---

### Requirement 8: Conv2d モデルアーキテクチャ（CoreML 互換）

**User Story:** As a iOS アプリ開発者, I want coremltools 3.4 / onnx-coreml 1.3 で正しく変換できる ONNX モデルが欲しい, so that Xcode で CoreML モデルとしてインポートしてオフライン推論に使用できる

#### Acceptance Criteria

1. THE BarkCNN2d SHALL 入力テンソル shape [B, 1, 40, 199] を受け取り、内部で [B, 40, 1, 199] に permute する
2. THE BarkCNN2d SHALL Conv2d(40, 64, (1,3), padding=(0,1)) → BatchNorm2d → ReLU → MaxPool2d((1,2)) を第 1 ブロックとして使用する
3. THE BarkCNN2d SHALL Conv2d(64, 128, (1,3), padding=(0,1)) → BatchNorm2d → ReLU → MaxPool2d((1,2)) を第 2 ブロックとして使用する
4. THE BarkCNN2d SHALL Conv2d(128, 128, (1,3), padding=(0,1)) → BatchNorm2d → ReLU を第 3 ブロックとして使用する
5. THE BarkCNN2d SHALL AdaptiveAvgPool2d((1,1)) で空間次元を集約し、Dropout → Linear(128, 1) → Sigmoid で出力 shape [B, 1] を生成する
6. THE BarkCNN2d SHALL 学習可能パラメータ数を 200,000 以下に抑える
7. THE BarkCNN2d SHALL dropout_rate パラメータ（デフォルト 0.3）を受け取り、Dropout レイヤーに適用する

---

### Requirement 9: Conv1d モデルアーキテクチャ（可変長推論）

**User Story:** As a 開発者, I want 可変長入力に対応した軽量モデルが欲しい, so that CLI で任意長の音声を効率的に推論できる

#### Acceptance Criteria

1. THE BarkCNN SHALL 入力テンソル shape [B, T, 40]（channels-last）を受け取り、内部で [B, 40, T]（channels-first）に permute する
2. THE BarkCNN SHALL 3 層の Conv1d ブロック（Conv1d → BatchNorm1d → ReLU → MaxPool1d）を持つ
3. THE BarkCNN SHALL Global Average Pooling（時間軸方向の mean）を使用して畳み込み出力を固定長ベクトルに集約する
4. THE BarkCNN SHALL Linear(128, 1) → Sigmoid で出力を 0.0〜1.0 の確率値とする
5. THE BarkCNN SHALL 可変長入力（任意の T）に対応し、動的軸付きで ONNX エクスポート可能とする

---

### Requirement 10: データ拡張

**User Story:** As a 開発者, I want 学習時にデータ拡張を適用したい, so that 限られた ESC-50 データセットでも汎化性能の高いモデルを学習できる

#### Acceptance Criteria

1. WHILE 学習モード（`is_train=True`）である場合, THE ESC50BarkDataset SHALL 各サンプルに対してタイムシフト（±1600 サンプル、16kHz で ±100ms の範囲で一様分布に従いランダムシフトし、範囲外はゼロパディング）を適用確率に基づいて適用する
2. WHILE 学習モード（`is_train=True`）である場合, THE ESC50BarkDataset SHALL 各サンプルに対してガウシアンノイズ付加（SNR 20〜40dB の範囲で一様分布に従い選択した SNR 値のノイズを加算）を適用確率に基づいて適用する
3. WHILE バリデーションモード（`is_train=False`）である場合, THE ESC50BarkDataset SHALL タイムシフトおよびガウシアンノイズ付加を一切適用せず、決定論的な前処理のみを実行する
4. THE ESC50BarkDataset SHALL データ拡張の適用確率を 0.0 以上 1.0 以下の範囲で設定可能とし、デフォルト値を 0.5 とする
5. THE ESC50BarkDataset SHALL タイムシフトとガウシアンノイズ付加を互いに独立して適用確率に基づき判定し、MFCC 抽出前の波形レベルで適用する

---

### Requirement 11: ONNX CoreML 互換性検証

**User Story:** As a 開発者, I want ONNX エクスポート後にモデルが CoreML 変換パイプラインの前提条件を満たすことを自動検証したい, so that 変換失敗を事前に検出できる

#### Acceptance Criteria

1. WHEN ONNX モデルがエクスポートされた場合, THE OnnxValidator SHALL 入力 shape が [1, 1, 40, 199] であることを検証する
2. WHEN ONNX モデルがエクスポートされた場合, THE OnnxValidator SHALL 出力 shape が [1, 1] であることを検証する
3. WHEN ONNX モデルがエクスポートされた場合, THE OnnxValidator SHALL opset_version が 9 であることを検証する
4. WHEN ONNX モデルがエクスポートされた場合, THE OnnxValidator SHALL 全次元が固定整数値（dim_value）であり、動的軸（dim_param）が存在しないことを検証する
5. WHEN ONNX モデルがエクスポートされた場合, THE OnnxValidator SHALL 使用オペレータが許可リスト（Conv, Relu, BatchNormalization, MaxPool, AveragePool, GlobalAveragePool, Reshape, Gemm, Sigmoid, Constant, Squeeze, Transpose）内であることを検証する
6. IF いずれかの検証項目が失敗した場合, THEN THE training pipeline SHALL 全検証項目を実行した上で、失敗項目ごとの理由を含むエラーメッセージを stderr に出力し、終了コード 1 で終了する
7. WHEN 全ての検証項目が成功した場合, THE training pipeline SHALL 検証合格を示すメッセージを stdout に出力する

---

### Requirement 12: BarkDetector モデル自動判別

**User Story:** As a 開発者, I want BarkDetector がモデル形式を自動判別して適切に推論したい, so that モデル切り替え時に CLI やコアライブラリの変更が不要になる

#### Acceptance Criteria

1. WHEN ONNX モデルの入力 shape が [1, 1, 40, N]（4D）の場合, THE BarkDetector SHALL CoreML 互換固定長モデルとして認識し、4D channels-first 推論を実行する
2. WHEN ONNX モデルの入力 shape が [1, 40, N]（3D、dim[1]=40）の場合, THE BarkDetector SHALL 固定長 channels-first モデルとして認識し、FeatureExtractor に fixed_length=N を指定して推論する
3. WHEN ONNX モデルの入力 shape が [1, T, 40]（3D、dim[2]=40）の場合, THE BarkDetector SHALL 可変長 channels-last モデルとして認識し、可変長前処理で推論する
4. IF ONNX モデルの入力 shape が上記いずれにも該当しない場合, THEN THE BarkDetector SHALL モデル形式を認識できないことを示すエラーを含む DetectionResult を返す
5. THE BarkDetector SHALL いずれのモデル形式を使用した場合でも、DetectionResult インターフェース（is_bark, confidence, timestamp, audio_duration, error）のフィールド構成および各フィールドの型を変更しない

---

### Requirement 13: UrbanSound8K dog_bark の正例利用

**User Story:** As a ML エンジニア, I want UrbanSound8K の dog_bark クラス（約 1,000 サンプル）を正例として学習に利用したい, so that 犬の吠え声検出モデルの汎化性能を向上させられる。

#### Acceptance Criteria

1. WHEN data_sources に "urbansound8k" が含まれる場合, THE UrbanSound8KBarkDataset SHALL UrbanSound8K メタデータ CSV（UrbanSound8K.csv）から classID=3 のエントリを正例（label=1）として読み込み、各エントリの音声ファイルパスを `urbansound8k_dir / audio / fold{fold} / {slice_file_name}` として解決する。
2. WHEN 読み込んだ音声のサンプル数が clip_duration_sec × sample_rate 未満の場合, THE BarkDatasetBase SHALL 音声末尾にゼロパディングを付加して clip_duration_sec × sample_rate サンプルの長さに揃える。
3. THE UrbanSound8KBarkDataset SHALL urbansound8k_val_fold（デフォルト: 10）で指定された fold 番号のエントリをバリデーション用、それ以外の fold（1〜10 のうち val_fold を除く）を学習用として分割する。

---

### Requirement 14: UrbanSound8K 暮らし系サウンドの負例利用

**User Story:** As a ML エンジニア, I want UrbanSound8K の暮らし系サウンド（children_playing, siren, car_horn, engine_idling）を負例として追加したい, so that 都市環境ノイズに対する偽陽性を低減できる。

#### Acceptance Criteria

1. WHEN data_sources に "urbansound8k" が含まれる場合, THE UrbanSound8KBarkDataset SHALL UrbanSound8K メタデータ CSV から urbansound8k_negative_classes で指定された classID のエントリを負例（label=0）として読み込む。
2. THE TrainingConfig SHALL urbansound8k_negative_classes フィールドを `list[int]` 型で定義し、デフォルト値として [2, 8, 1, 5]（children_playing, siren, car_horn, engine_idling）を保持する。
3. THE TrainingConfig SHALL urbansound8k_positive_classes フィールドを `list[int]` 型で定義し、デフォルト値として [3]（dog_bark）を保持する。
4. IF urbansound8k_positive_classes と urbansound8k_negative_classes に同一の classID が含まれる場合, THEN THE DatasetFactory SHALL 重複する classID を示すエラーメッセージを出力し、データセット構築を中断する。
5. IF urbansound8k_negative_classes に 0〜9 の範囲外の classID が含まれる場合, THEN THE DatasetFactory SHALL 無効な classID 値を示すエラーメッセージを出力し、データセット構築を中断する。

---

### Requirement 15: Dataset 抽象基底クラスの導入

**User Story:** As a 開発者, I want Dataset クラスを抽象化して複数データソースを統一的に扱いたい, so that 将来のデータソース追加時にも最小限の変更で対応できる。

#### Acceptance Criteria

1. THE BarkDatasetBase SHALL torch.utils.data.Dataset を継承した抽象基底クラスとして、共通処理（ランダムクロップ、MFCC 抽出、データ拡張、ゼロパディング）を `__getitem__` 内に実装する。
2. THE BarkDatasetBase SHALL 抽象メソッド load_entries() を定義し、サブクラスに `list[dict]` 形式のエントリリスト返却を要求する。
3. FOR ALL load_entries() が返すエントリ, THE BarkDatasetBase SHALL 各エントリが "filepath"（音声ファイルの Path）、"label"（0 または 1 の整数）、"fold"（1 以上の正の整数）のキーを含むことを前提とする。
4. WHEN model_type が "conv2d" の場合, THE BarkDatasetBase の `__getitem__` SHALL shape [1, 40, fixed_frame_length] の float32 テンソルとラベル shape [1] のタプルを返す。WHEN model_type が "conv1d" の場合, THE BarkDatasetBase の `__getitem__` SHALL shape [T, 40] の float32 テンソルとラベル shape [1] のタプルを返す。
5. WHILE is_train が True の場合, THE BarkDatasetBase SHALL ランダムクロップとデータ拡張（model_type が "conv2d" の場合のみ）を適用する。WHILE is_train が False の場合, THE BarkDatasetBase SHALL 中央クロップを適用し、データ拡張を適用しない。
6. IF load_entries() が返すエントリの "filepath" が存在しないファイルを指す場合, THEN THE BarkDatasetBase SHALL `__getitem__` 呼び出し時にファイルが見つからないことを示す例外を発生させる。
7. THE ESC50BarkDataset SHALL BarkDatasetBase を継承し、ESC-50 固有のメタデータ読み込みを load_entries() で実装する。

---

### Requirement 16: TrainingConfig のデータソース指定

**User Story:** As a ML エンジニア, I want TrainingConfig で利用するデータソースを柔軟に指定したい, so that 実験ごとに異なるデータソースの組み合わせで学習を行える。

#### Acceptance Criteria

1. THE TrainingConfig SHALL data_sources フィールド（型: `list[str]`）を持ち、デフォルト値として `["esc50"]` を設定する。
2. THE TrainingConfig SHALL urbansound8k_dir フィールド（型: `Path`）を持ち、デフォルト値として `Path("data/UrbanSound8K")` を設定する。
3. THE TrainingConfig SHALL urbansound8k_val_fold フィールド（型: `int`、有効範囲: 1〜10）を持ち、デフォルト値として 10 を設定する。
4. WHEN CLI 引数 --data-sources が指定された場合, THE 学習スクリプト SHALL カンマ区切りの文字列をパースし、各要素の前後空白を除去した上で data_sources フィールドに設定する（例: "esc50,urbansound8k"）。
5. IF data_sources に "esc50" および "urbansound8k" 以外の文字列が含まれる場合, THEN THE 学習スクリプト SHALL 無効なデータソース名を示すエラーメッセージを出力し、有効なデータソース名の一覧を表示して終了する。

---

### Requirement 17: Dataset ファクトリと ConcatDataset 結合

**User Story:** As a 開発者, I want ファクトリ関数で設定に基づいた Dataset を自動構築したい, so that データソースの組み合わせロジックを呼び出し側から分離できる。

#### Acceptance Criteria

1. THE DatasetFactory SHALL build_dataset(config, is_train) 関数を提供し、TrainingConfig の data_sources の各要素に対応する Dataset インスタンスを構築する。
2. WHEN data_sources に複数のデータソースが指定された場合, THE DatasetFactory SHALL ConcatDataset を使用して各データソースの Dataset を結合し、結合後の Dataset を返す。
3. WHEN is_train=True かつ結合後（または単一）のデータセットに正例と負例の両方が存在する場合, THE DatasetFactory SHALL 正例を負例の件数と一致するようオーバーサンプリング（repeat_factor + remainder 方式）してクラスバランスを調整する。
4. WHEN is_train=False の場合, THE DatasetFactory SHALL クラスバランス調整を適用せず、データソースから取得したバリデーション用エントリをそのまま返す。
5. WHEN data_sources に単一のデータソースのみが指定された場合, THE DatasetFactory SHALL ConcatDataset を使用せず該当する単一の Dataset インスタンスを返す。
6. IF data_sources に "esc50" および "urbansound8k" 以外の未知のデータソース名が含まれる場合, THEN THE DatasetFactory SHALL 未対応のデータソース名を示すエラーを発生させる。

---

### Requirement 18: 後方互換性の維持

**User Story:** As a 既存ユーザー, I want ESC-50 のみでの学習が引き続き動作することを保証したい, so that UrbanSound8K を未ダウンロードの環境でも学習パイプラインが利用できる。

#### Acceptance Criteria

1. WHEN data_sources がデフォルト値 `["esc50"]` のまま使用された場合, THE 学習パイプライン SHALL ESC-50 のみを使用して学習を実行し、UrbanSound8K のファイル読み込み・ディレクトリ参照・メタデータ CSV パースのいずれも行わない。
2. WHEN --data-sources CLI 引数が省略された場合, THE 学習スクリプト SHALL data_sources のデフォルト値 `["esc50"]` を使用する。
3. THE ESC50BarkDataset SHALL BarkDatasetBase を継承した後も、コンストラクタ引数として TrainingConfig と is_train（bool）を受け取るインタフェースを維持し、ESC-50 の fold 分割（val_fold: 1〜5）と正例クラス（positive_classes）・負例クラス（negative_classes）の指定に基づくフィルタリング動作を維持する。

---

### Requirement 19: UrbanSound8K 手動ダウンロード案内

**User Story:** As a ML エンジニア, I want UrbanSound8K が未ダウンロード時に案内メッセージを表示してほしい, so that ライセンス上の制約を理解した上で手動でダウンロードできる。

#### Acceptance Criteria

1. IF data_sources に "urbansound8k" が含まれるが urbansound8k_dir/metadata/UrbanSound8K.csv が存在しない場合, THEN THE DatasetFactory SHALL FileNotFoundError を raise し、エラーメッセージにダウンロード先 URL（https://urbansounddataset.weebly.com/urbansound8k.html）および期待するディレクトリ配置構成（metadata/UrbanSound8K.csv と audio/fold1〜fold10）を含める。
2. IF urbansound8k_dir/metadata/UrbanSound8K.csv が存在しない場合に FileNotFoundError が raise された場合, THEN THE DatasetFactory SHALL 他の data_sources（例: "esc50"）にフォールバックせず、処理を中断する。
3. THE DatasetFactory SHALL UrbanSound8K の自動ダウンロードを行わない（Creative Commons Attribution Non-Commercial 4.0 ライセンスの制約により手動ダウンロードのみ対応）。
