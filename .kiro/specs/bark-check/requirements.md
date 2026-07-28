# Requirements Document

## Introduction

bark-check は、音声ファイルを入力として受け取り、その音声が犬の吠え声（bark）を含むかどうかを判定するツールである。
Python CLI コマンドとして提供され、判定ロジックは iOS Swift アプリなど他のプラットフォームへの転用を考慮したポータブルな設計とする。

判定ロジックはコアライブラリ（BarkDetector）として独立させ、CLI はそのラッパーとして実装する。
BarkDetector は音声ファイルを一切扱わず、メモリ上のモノラル PCM サンプル列（float32 配列）とサンプリングレート（Hz）のみを入力として受け付ける。
これにより、1〜2 秒程度の音声ブロックをリアルタイムで判定するユースケース（iOS Swift アプリなど）に対しても、同一のアルゴリズムを転用できる。

CLI レイヤーでは AudioLoader が音声ファイルをデコードしてモノラル PCM に変換し、BarkDetector に渡す。

## Glossary

- **BarkDetector**: モノラル PCM サンプル列を入力として受け取り、犬の吠え声を検出するコアロジックモジュール
- **PcmBlock**: モノラル PCM サンプルの配列。float32 型、サンプリングレートは任意だが 16kHz を推奨
- **AudioLoader**: 音声ファイルをデコードしてモノラル PCM に変換する CLI レイヤーのモジュール
- **DetectionResult**: 判定結果を表すデータ構造。吠え声の有無（boolean）、信頼度スコア（0.0〜1.0）、タイムスタンプ、音声長、エラー情報を含む
- **CLI**: `bark-check` コマンドとして提供されるコマンドラインインターフェース
- **ConfidenceScore**: BarkDetector が出力する予測の確からしさを示す 0.0 以上 1.0 以下の浮動小数点数
- **FeatureExtractor**: PcmBlock から機械学習モデルへの入力特徴量（例: MFCC）を抽出するモジュール

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
