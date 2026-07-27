# Requirements Document

## Introduction

BarkDetector の入力長上限チェックが「サンプル数 32,000 固定」で実装されているため、16kHz 以外のサンプリングレート（44.1kHz, 48kHz 等）の入力において、実時間で 2 秒未満の有効な音声でも不正にエラーが返されるバグを修正する。

修正方針として、定数 `_MAX_PCM_LENGTH = 32000` を `_MAX_DURATION_SEC = 10.0` に置き換え、上限チェックを `len(pcm) / sample_rate > 10.0` に変更する。これにより、サンプリングレートに依存しない時間長ベースの制限を実現する。

## Glossary

- **BarkDetector**: モノラル PCM サンプル列を入力として受け取り、犬の吠え声を検出するコアロジックモジュール
- **DetectionResult**: 判定結果を表すデータ構造。`is_bark`（真偽値）、`confidence`（0.0〜1.0）、`timestamp`、`audio_duration`、`error` を含む
- **Bug Condition C(X)**: `len(pcm) > 32000 AND len(pcm) / sample_rate <= 10.0` を満たす入力。修正前コードでは不正にエラーが返される
- **`_MAX_PCM_LENGTH`**: 修正前の定数。値は 32,000（サンプル数）
- **`_MAX_DURATION_SEC`**: 修正後の定数。値は 10.0（秒）

---

## Requirements

### Requirement 1: 時間長ベースの入力上限チェック

**User Story:** As a 開発者, I want BarkDetector が入力長をサンプリングレートに依存しない時間長で制限してほしい, so that 任意のサンプリングレートの有効な音声を正しく処理できる

#### Acceptance Criteria

1.1 WHEN 任意のサンプリングレートの入力で `len(pcm) / sample_rate <= 10.0` の場合, THE BarkDetector SHALL 入力長に関するエラーを返さずに推論処理を実行する

1.2 WHEN 任意のサンプリングレートの入力で `len(pcm) / sample_rate > 10.0` の場合, THE BarkDetector SHALL `is_bark=False`, `confidence=0.0`, `error="Input exceeds maximum duration of 10.0 seconds"` を含む DetectionResult を返す

1.3 WHEN `len(pcm) / sample_rate` がちょうど 10.0 の場合, THE BarkDetector SHALL 入力長エラーを返さずに推論処理を実行する

1.4 THE BarkDetector SHALL 上限チェックの判定基準として `len(pcm) / sample_rate > _MAX_DURATION_SEC` を使用し、サンプル数による固定値比較を行わない

---

### Requirement 2: 上限超過時のエラーメッセージ

**User Story:** As a 開発者, I want 上限超過時のエラーメッセージから実際の制限事項（時間長）を把握したい, so that 入力を適切に調整できる

#### Acceptance Criteria

2.1 WHEN 入力が 10 秒を超過した場合, THE BarkDetector SHALL エラーメッセージとして `"Input exceeds maximum duration of 10.0 seconds"` を返す

2.2 THE BarkDetector SHALL サンプル数ベースのエラーメッセージ `"Input exceeds maximum length of 32000 samples"` を返さない

---

### Requirement 3: サンプリングレート非依存の正常処理

**User Story:** As a 開発者, I want 10 秒以内の音声をサンプリングレートに関係なく処理したい, so that 44.1kHz や 48kHz の音声でも正しく推論できる

#### Acceptance Criteria

3.1 WHEN サンプリングレートが 44100Hz で `len(pcm) / 44100 <= 10.0` の入力が与えられた場合, THE BarkDetector SHALL 入力長エラーを返さずに推論処理を実行する

3.2 WHEN サンプリングレートが 48000Hz で `len(pcm) / 48000 <= 10.0` の入力が与えられた場合, THE BarkDetector SHALL 入力長エラーを返さずに推論処理を実行する

3.3 WHEN サンプリングレートが 16000Hz で `len(pcm) / 16000 <= 10.0` の入力が与えられた場合, THE BarkDetector SHALL 入力長エラーを返さずに推論処理を実行する

3.4 WHEN サンプリングレートが 8000〜48000Hz の範囲の任意の値で `len(pcm) / sample_rate <= 10.0` の入力が与えられた場合, THE BarkDetector SHALL 入力長エラーを返さずに推論処理を実行する

---

### Requirement 4: 既存動作の保持（Regression Prevention）

**User Story:** As a 開発者, I want 既存の正常動作が修正によって壊れないことを保証したい, so that バグ修正が新たなバグを生まない

#### Acceptance Criteria

4.1 WHEN 入力が空（サンプル数ゼロ）の場合, THE BarkDetector SHALL CONTINUE TO `"Input PCM block is empty"` エラーを含む DetectionResult を返す

4.2 WHEN 入力が無音（全サンプルがゼロ）かつ 10 秒以内の場合, THE BarkDetector SHALL CONTINUE TO `is_bark=False`, `confidence=0.0`, `error=None` の DetectionResult を返す

4.3 WHEN モデルが未ロード状態の場合, THE BarkDetector SHALL CONTINUE TO `"No model loaded"` エラーを含む DetectionResult を返す

4.4 WHEN 推論中にランタイムエラーが発生した場合, THE BarkDetector SHALL CONTINUE TO `"Inference error: <message>"` エラーを含む DetectionResult を返し例外を伝播しない

4.5 WHEN threshold が [0.0, 1.0] 範囲外で BarkDetector を初期化しようとした場合, THE BarkDetector SHALL CONTINUE TO ValueError を送出する

4.6 WHEN 10 秒以内の有効な入力が与えられた場合, THE BarkDetector SHALL CONTINUE TO 入力全長を FeatureExtractor → モデルに渡して推論結果を返す（クロップしない）

---

### Requirement 5: Correctness Properties

**User Story:** As a 開発者, I want バグ修正の正しさをプロパティベーステストで検証したい, so that 全入力空間にわたってバグが修正されたことと既存動作が保持されていることを保証できる

#### Acceptance Criteria

5.1 FOR ALL inputs WHERE `len(pcm) > 32000 AND len(pcm) / sample_rate <= 10.0` (Bug Condition), THE fixed BarkDetector SHALL NOT return an error containing "maximum length" or "maximum duration"

5.2 FOR ALL inputs WHERE NOT Bug Condition (i.e., `len(pcm) <= 32000` OR `len(pcm) / sample_rate > 10.0` OR `len(pcm) == 0`), THE fixed BarkDetector SHALL produce the same DetectionResult as the original code

5.3 FOR ALL inputs WHERE `len(pcm) / sample_rate > 10.0`, THE fixed BarkDetector SHALL return `error="Input exceeds maximum duration of 10.0 seconds"`, `is_bark=False`, `confidence=0.0`
