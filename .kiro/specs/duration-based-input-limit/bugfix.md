# Bugfix Requirements Document

## Introduction

BarkDetector の入力長上限チェックが「サンプル数ベース（32,000 samples）」で実装されているため、16kHz 以外のサンプリングレート（例: 44.1kHz）の入力において、2 秒未満の有効な音声でもサンプル数が 32,000 を超えて不正にエラーが返される。

上限の目的は極端に長い入力を弾く安全弁であり、学習モデルの最適入力長を強制するものではない。モデルは Global Average Pooling を使用しており任意長の入力を処理可能であるため、上限を「10 秒」の時間長ベースに変更し、サンプリングレートに依存しない正しい判定を行う。

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN サンプリングレートが 16kHz でない入力（例: 44.1kHz）で `len(pcm) > 32000` かつ実際の音声長が 2 秒未満の場合 THEN the system は `"Input exceeds maximum length of 32000 samples"` エラーを返し、有効な音声の推論を拒否する

1.2 WHEN サンプリングレートが 16kHz の入力で `len(pcm) > 32000`（2 秒超）かつ実際の音声長が 10 秒以内の場合 THEN the system は `"Input exceeds maximum length of 32000 samples"` エラーを返し、安全な長さの音声の推論を拒否する

1.3 WHEN 上限超過エラーが発生した場合 THEN the system はサンプル数ベースのエラーメッセージ `"Input exceeds maximum length of 32000 samples"` を返し、実際の制限事項（時間長）を正確に伝えない

### Expected Behavior (Correct)

2.1 WHEN 任意のサンプリングレートの入力で `len(pcm) / sample_rate <= 10.0`（10 秒以内）の場合 THEN the system SHALL 入力長エラーを返さずに推論処理を実行する

2.2 WHEN 任意のサンプリングレートの入力で `len(pcm) / sample_rate > 10.0`（10 秒超）の場合 THEN the system SHALL `"Input exceeds maximum duration of 10.0 seconds"` エラーを含む DetectionResult を返す

2.3 WHEN 上限超過エラーが発生した場合 THEN the system SHALL 時間長ベースのエラーメッセージ `"Input exceeds maximum duration of 10.0 seconds"` を返し、制限事項を正確に伝える

### Unchanged Behavior (Regression Prevention)

3.1 WHEN 入力が空（サンプル数ゼロ）の場合 THEN the system SHALL CONTINUE TO `"Input PCM block is empty"` エラーを含む DetectionResult を返す

3.2 WHEN 入力が無音（全サンプルがゼロ）かつ 10 秒以内の場合 THEN the system SHALL CONTINUE TO `is_bark=False`, `confidence=0.0`, `error=None` の DetectionResult を返す

3.3 WHEN モデルが未ロード状態の場合 THEN the system SHALL CONTINUE TO `"No model loaded"` エラーを含む DetectionResult を返す

3.4 WHEN 推論中にランタイムエラーが発生した場合 THEN the system SHALL CONTINUE TO `"Inference error: <message>"` エラーを含む DetectionResult を返し例外を伝播しない

3.5 WHEN threshold が [0.0, 1.0] 範囲外で BarkDetector を初期化しようとした場合 THEN the system SHALL CONTINUE TO ValueError を送出する

3.6 WHEN 10 秒以内の有効な入力が与えられた場合 THEN the system SHALL CONTINUE TO 全長を FeatureExtractor → モデルに渡して推論結果を返す（クロップしない）

---

### Bug Condition（構造化擬似コード）

```pascal
FUNCTION isBugCondition(X)
  INPUT: X of type (pcm: float32[], sample_rate: int)
  OUTPUT: boolean

  // サンプル数が 32,000 を超えるが、時間長は 10 秒以内の入力
  RETURN len(X.pcm) > 32000 AND len(X.pcm) / X.sample_rate <= 10.0
END FUNCTION
```

### Fix Checking（修正確認プロパティ）

```pascal
// Property: Fix Checking - 10 秒以内の入力はエラーなしに推論される
FOR ALL X WHERE isBugCondition(X) DO
  result ← BarkDetector'.detect(X.pcm, X.sample_rate)
  ASSERT result.error != "Input exceeds maximum length of 32000 samples"
  ASSERT result.error IS NULL OR result.error DOES NOT CONTAIN "maximum length"
END FOR
```

### Preservation Checking（保存確認プロパティ）

```pascal
// Property: Preservation Checking - 非バグ条件の入力は修正前後で同一動作
FOR ALL X WHERE NOT isBugCondition(X) DO
  // 空入力・無音入力・モデル未ロード・正常推論のいずれも動作不変
  ASSERT BarkDetector(X.pcm, X.sample_rate) = BarkDetector'(X.pcm, X.sample_rate)
END FOR
```

### New Behavior（新規上限チェック）

```pascal
// Property: 10 秒超の入力はエラー DetectionResult を返す
FOR ALL X WHERE len(X.pcm) / X.sample_rate > 10.0 DO
  result ← BarkDetector'.detect(X.pcm, X.sample_rate)
  ASSERT result.error = "Input exceeds maximum duration of 10.0 seconds"
  ASSERT result.is_bark = False
  ASSERT result.confidence = 0.0
END FOR
```
