# Duration-Based Input Limit Bugfix Design

## Overview

BarkDetector の入力長上限チェックが「サンプル数ベース（`_MAX_PCM_LENGTH = 32000`）」で実装されているため、16kHz 以外のサンプリングレート（例: 44.1kHz, 48kHz）の入力において、実時間で 2 秒未満の有効な音声でも不正にエラーが返されるバグを修正する。

修正方針として、定数 `_MAX_PCM_LENGTH = 32000` を `_MAX_DURATION_SEC = 10.0` に置き換え、上限チェックを `len(pcm) / sample_rate > 10.0` に変更する。これにより、サンプリングレートに依存しない時間長ベースの制限を実現し、モデルの Global Average Pooling による任意長入力対応能力を活かす。

## Glossary

- **Bug_Condition (C)**: サンプル数が 32,000 を超えるが、実時間長が 10 秒以内の入力。現在のコードはこれをエラーとして拒否するが、本来は推論可能な有効入力である
- **Property (P)**: バグ条件に該当する入力に対して、入力長エラーを返さずに正常に推論処理を実行すること
- **Preservation**: 空入力チェック、無音入力チェック、モデル未ロードチェック、推論エラー処理、閾値バリデーション等の既存動作が修正前後で不変であること
- **`_MAX_PCM_LENGTH`**: 現在の定数。値は 32,000（サンプル数）。`bark_detector.py` のモジュールレベルで定義
- **`_MAX_DURATION_SEC`**: 修正後の定数。値は 10.0（秒）。サンプリングレートと組み合わせて上限を判定する
- **`BarkDetector.detect()`**: `bark_detector.py` の主メソッド。PCM 配列とサンプリングレートを受け取り DetectionResult を返す
- **Global Average Pooling**: モデルアーキテクチャに採用されているプーリング手法。入力長に依存せず任意長のテンソルを処理可能

## Bug Details

### Bug Condition

バグは、サンプリングレートが 16kHz でない入力（特に 44.1kHz や 48kHz）を処理する際、あるいは 16kHz であっても 2 秒を超え 10 秒以内の音声を渡した際に発生する。`detect()` 内の上限チェックが `len(pcm) > _MAX_PCM_LENGTH`（32,000 サンプル固定）で行われているため、実時間長が安全な範囲内でもサンプル数だけで拒否される。

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (pcm: float32[], sample_rate: int)
  OUTPUT: boolean

  // サンプル数が 32,000 を超えるが、時間長は 10 秒以内の入力
  RETURN len(input.pcm) > 32000
         AND len(input.pcm) / input.sample_rate <= 10.0
END FUNCTION
```

### Examples

- **44.1kHz, 0.8 秒の音声**: `len(pcm) = 35,280` → 32,000 超でエラーになるが、実時間は 0.8 秒で有効
- **48kHz, 0.7 秒の音声**: `len(pcm) = 33,600` → 32,000 超でエラーになるが、実時間は 0.7 秒で有効
- **16kHz, 5 秒の音声**: `len(pcm) = 80,000` → 32,000 超でエラーになるが、実時間は 5 秒で有効（10 秒以内）
- **44.1kHz, 11 秒の音声**: `len(pcm) = 485,100` → 10 秒超なので修正後もエラーを返す（正しい動作）

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- 空入力（`len(pcm) == 0`）に対して `"Input PCM block is empty"` エラーを返す動作
- 無音入力（全サンプルがゼロ）かつ 10 秒以内に対して `is_bark=False`, `confidence=0.0`, `error=None` を返す動作
- モデル未ロード（`_session is None`）時に `"No model loaded"` エラーを返す動作
- 推論中ランタイムエラー時に `"Inference error: <message>"` エラーを返し例外を伝播しない動作
- `threshold` が [0.0, 1.0] 範囲外の場合に `ValueError` を送出する動作
- 有効な入力を全長そのまま FeatureExtractor → モデルに渡す動作（クロップしない）

**Scope:**
上限チェックの条件変更のみが対象であり、以下に該当する入力は修正の影響を受けない:
- 空入力（サンプル数 0）
- 無音入力（全ゼロ）で 10 秒以内
- サンプル数が 32,000 以下の入力（修正前後どちらでもエラーにならない）
- 時間長が 10 秒を超える入力（修正前後どちらでもエラーになる）

## Hypothesized Root Cause

バグの根本原因は明確である:

1. **サンプル数ベースの定数設計**: `_MAX_PCM_LENGTH = 32000` は「16kHz × 2 秒 = 32,000 サンプル」という前提で設計されたが、BarkDetector のインターフェースは任意のサンプリングレートを受け付ける設計である。この前提とインターフェースの不整合がバグの直接原因
2. **モデル制約の誤解**: 当初はモデルの入力テンソル長が固定と想定されていた可能性があるが、実際のモデルは Global Average Pooling を使用しており任意長入力に対応可能
3. **安全マージンの不足**: 元の制限（2 秒）は短すぎ、実用的な音声（犬の吠え声は数秒にわたることがある）を不正に拒否する。10 秒への緩和が適切

## Correctness Properties

Property 1: Bug Condition - 10 秒以内の入力は入力長エラーを返さない

_For any_ input where the bug condition holds (サンプル数 > 32,000 かつ `len(pcm) / sample_rate <= 10.0`), the fixed `BarkDetector.detect()` SHALL 入力長に関するエラーを返さず、推論処理を実行する（エラーが返る場合は "No model loaded" や "Inference error:" 等の別原因に限られる）。

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation - 非バグ条件の入力は修正前後で同一動作

_For any_ input where the bug condition does NOT hold (サンプル数 <= 32,000、または時間長 > 10 秒、または空入力、または無音入力), the fixed code SHALL produce the same result as the original code, preserving 空入力チェック・無音チェック・モデル未ロードチェック・推論エラー処理・閾値判定の全動作。

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

Property 3: New Behavior - 10 秒超の入力は時間長ベースのエラーを返す

_For any_ input where `len(pcm) / sample_rate > 10.0`, the fixed `BarkDetector.detect()` SHALL `error = "Input exceeds maximum duration of 10.0 seconds"`, `is_bark = False`, `confidence = 0.0` を含む DetectionResult を返す。

**Validates: Requirements 2.2, 2.3**

## Fix Implementation

### Changes Required

修正は最小限かつ明確である。

**File**: `src/bark_check/bark_detector.py`

**Specific Changes**:

1. **定数の変更**: `_MAX_PCM_LENGTH = 32000` を `_MAX_DURATION_SEC = 10.0` に置き換える
   ```python
   # Before
   _MAX_PCM_LENGTH = 32000
   
   # After
   _MAX_DURATION_SEC = 10.0
   ```

2. **上限チェック条件の変更**: `len(pcm) > _MAX_PCM_LENGTH` を `len(pcm) / sample_rate > _MAX_DURATION_SEC` に変更する
   ```python
   # Before
   if len(pcm) > _MAX_PCM_LENGTH:
       return DetectionResult(
           is_bark=False,
           confidence=0.0,
           timestamp=timestamp,
           audio_duration=audio_duration,
           error="Input exceeds maximum length of 32000 samples",
       )
   
   # After
   if len(pcm) / sample_rate > _MAX_DURATION_SEC:
       return DetectionResult(
           is_bark=False,
           confidence=0.0,
           timestamp=timestamp,
           audio_duration=audio_duration,
           error="Input exceeds maximum duration of 10.0 seconds",
       )
   ```

3. **エラーメッセージの変更**: サンプル数ベースのメッセージから時間長ベースのメッセージに更新する
   - Before: `"Input exceeds maximum length of 32000 samples"`
   - After: `"Input exceeds maximum duration of 10.0 seconds"`

**File**: `tests/test_bark_detector.py`

**Specific Changes**:

4. **ユニットテスト修正**: `test_上限超過入力のとき_error_が返ること` を時間長ベースの検証に更新する

5. **Property 3 テスト修正**: `test_property_oversized_input_returns_error` のストラテジーを時間長 > 10 秒の入力を生成するように変更し、エラーメッセージの検証も更新する

## Testing Strategy

### Validation Approach

テスト戦略は二段階アプローチに従う: まず修正前コードでバグを再現するカウンターエクザンプルを生成し、次に修正後コードで正しい動作と既存動作の保持を検証する。

### Exploratory Bug Condition Checking

**Goal**: 修正前コードでバグを再現し、根本原因を確認する。根本原因を否定する場合は再分析が必要。

**Test Plan**: 様々なサンプリングレートで 10 秒以内かつ 32,000 サンプル超の入力を生成し、修正前コードで不正にエラーが返ることを確認する。

**Test Cases**:
1. **44.1kHz 0.8 秒入力テスト**: `len(pcm) = 35,280` を渡してエラーが返ることを確認（修正前コードで失敗）
2. **48kHz 0.7 秒入力テスト**: `len(pcm) = 33,600` を渡してエラーが返ることを確認（修正前コードで失敗）
3. **16kHz 5 秒入力テスト**: `len(pcm) = 80,000` を渡してエラーが返ることを確認（修正前コードで失敗）
4. **22.05kHz 1.5 秒入力テスト**: `len(pcm) = 33,075` を渡してエラーが返ることを確認（修正前コードで失敗）

**Expected Counterexamples**:
- 全てのケースで `"Input exceeds maximum length of 32000 samples"` エラーが返される
- 原因: `len(pcm) > 32000` の固定条件によりサンプリングレートが考慮されていない

### Fix Checking

**Goal**: バグ条件に該当する全入力に対して、修正後の関数が入力長エラーを返さないことを検証する。

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := BarkDetector_fixed.detect(input.pcm, input.sample_rate)
  ASSERT result.error != "Input exceeds maximum length of 32000 samples"
  ASSERT result.error IS NULL OR NOT CONTAINS(result.error, "maximum duration")
  // エラーがある場合は "No model loaded" 等の別原因に限られる
END FOR
```

### Preservation Checking

**Goal**: バグ条件に該当しない全入力に対して、修正後の関数が修正前と同一の結果を返すことを検証する。

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT BarkDetector_original.detect(input.pcm, input.sample_rate)
       = BarkDetector_fixed.detect(input.pcm, input.sample_rate)
END FOR
```

**Testing Approach**: プロパティベーステスト（Hypothesis）を用いた保存確認を推奨する。理由:
- 入力空間全体にわたって自動的に多数のテストケースを生成できる
- 手動ユニットテストで見逃しがちなエッジケースを捕捉できる
- 非バグ条件の入力に対する動作不変性を高い信頼度で保証できる

**Test Plan**: 修正前コードで正常動作する入力（空入力、無音入力、32,000 サンプル以下の入力、10 秒超の入力）の動作を観察し、修正後もこれらの動作が保持されることをプロパティベーステストで検証する。

**Test Cases**:
1. **空入力の保存**: 修正前後で `"Input PCM block is empty"` エラーが同一であることを検証
2. **無音入力の保存**: 修正前後で `is_bark=False`, `confidence=0.0`, `error=None` が同一であることを検証
3. **32,000 サンプル以下入力の保存**: 修正前後で同一の DetectionResult が返ることを検証
4. **10 秒超入力のエラー**: 修正後で `"Input exceeds maximum duration of 10.0 seconds"` エラーが返ることを検証

### Unit Tests

- 44.1kHz, 48kHz, 22.05kHz 等の各サンプリングレートで 10 秒以内の入力が正常処理されることのテスト
- 境界値テスト: ちょうど 10 秒の入力（`len(pcm) / sample_rate == 10.0`）がエラーにならないことの確認
- 境界値テスト: 10 秒をわずかに超える入力がエラーになることの確認
- エラーメッセージが `"Input exceeds maximum duration of 10.0 seconds"` であることの確認
- 空入力・無音入力のテストが修正後も通過することの確認

### Property-Based Tests

- ランダムなサンプリングレート (8,000〜48,000 Hz) と 10 秒以内の長さの PCM を生成し、入力長エラーが返されないことを検証（Property 1: Fix Checking）
- ランダムな空入力・無音入力・32,000 サンプル以下の入力を生成し、修正前後で同一動作であることを検証（Property 2: Preservation）
- ランダムなサンプリングレートで 10 秒超の PCM を生成し、正しいエラーメッセージが返されることを検証（Property 3: New Behavior）

### Integration Tests

- 実際の 44.1kHz WAV ファイル（10 秒以内）を CLI に渡し、エラーなく推論結果が返ることの確認
- 10 秒超の長い音声ファイルを CLI に渡し、適切なエラーメッセージが表示されることの確認
- 既存の 16kHz テストファイルが修正後も正常に動作することの確認
