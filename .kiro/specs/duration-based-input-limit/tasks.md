# Implementation Plan

- [x] 1. バグ条件探索テストを作成する
  - **Property 1: Bug Condition** - サンプル数 > 32,000 かつ時間長 <= 10 秒の入力が不正にエラーを返す
  - **CRITICAL**: このテストは修正前コードで FAIL することが期待される - 失敗がバグの存在を証明する
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: このテストは期待動作をエンコードする - 修正後に PASS することでバグ修正を検証する
  - **GOAL**: バグ条件に該当する入力でカウンターエクザンプルを生成し、バグの存在を実証する
  - **Scoped PBT Approach**: サンプリングレート 8,000〜48,000 Hz、時間長 0.1〜10.0 秒でサンプル数が 32,000 を超える入力を生成する
  - Hypothesis ストラテジー: `sample_rate` を `st.integers(8000, 48000)` で生成し、`duration` を `st.floats(0.1, 10.0)` で生成、`len(pcm) = int(sample_rate * duration)` が 32,000 を超える組み合わせに限定
  - Bug Condition: `len(pcm) > 32000 AND len(pcm) / sample_rate <= 10.0` (design.md の isBugCondition)
  - テストアサーション: `result.error` が `None` であること、または入力長に関するエラーでないこと（Expected Behavior: 入力長エラーを返さずに推論処理を実行する）
  - 修正前コードで実行 → テストが FAIL する（`"Input exceeds maximum length of 32000 samples"` エラーが返されるため）
  - カウンターエクザンプル例: `44.1kHz, 0.8秒 → len=35,280 → エラー`、`48kHz, 0.7秒 → len=33,600 → エラー`、`16kHz, 5秒 → len=80,000 → エラー`
  - テスト完了条件: テストが作成され、修正前コードで実行して FAIL が確認・文書化されたこと
  - _Requirements: 1.1, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 5.1_

- [x] 2. 保存プロパティテストを作成する（修正実装の前に）
  - **Property 2: Preservation** - 非バグ条件の入力は修正前後で同一動作を維持する
  - **IMPORTANT**: 観察優先方法論に従うこと
  - **観察フェーズ**: 修正前コードで以下の非バグ条件入力の動作を観察する
    - 空入力: `len(pcm) == 0` → `error="Input PCM block is empty"` を確認
    - 無音入力（全ゼロ、32,000 サンプル以下）: `is_bark=False`, `confidence=0.0`, `error=None` を確認
    - 32,000 サンプル以下の有効入力: モデル未ロード時は `error="No model loaded"` を確認
    - 10 秒超の入力（修正前は 32,000 超でエラー、修正後は時間長でエラー）: 動作変更が適切であることを確認
  - **プロパティベーステスト作成**: 非バグ条件の入力ドメイン全体にわたるプロパティテストを作成
    - 空入力プロパティ: 空配列 → 常に `error="Input PCM block is empty"` であること
    - 無音入力プロパティ: 全ゼロ配列（1〜32,000 サンプル）→ 常に `is_bark=False`, `confidence=0.0`, `error=None` であること
    - 短い入力プロパティ: `len(pcm) <= 32000` の非ゼロ入力 → 入力長エラーが返されないこと
    - 閾値バリデーションプロパティ: 範囲外の threshold → `ValueError` が送出されること
  - 修正前コードで実行 → テストが PASS する（保存すべき動作のベースラインを確認）
  - テスト完了条件: テストが作成され、修正前コードで実行して全て PASS が確認されたこと
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.2_

- [x] 3. 時間長ベース入力上限チェックの修正

  - [x] 3.1 `bark_detector.py` の定数・上限チェック条件を変更する
    - `_MAX_PCM_LENGTH = 32000` を `_MAX_DURATION_SEC = 10.0` に置き換える
    - 上限チェック条件を `len(pcm) > _MAX_PCM_LENGTH` から `len(pcm) / sample_rate > _MAX_DURATION_SEC` に変更する
    - エラーメッセージを `"Input exceeds maximum length of 32000 samples"` から `"Input exceeds maximum duration of 10.0 seconds"` に変更する
    - _Bug_Condition: isBugCondition(input) where len(input.pcm) > 32000 AND len(input.pcm) / input.sample_rate <= 10.0_
    - _Expected_Behavior: バグ条件に該当する入力に対して入力長エラーを返さず、推論処理を実行する_
    - _Preservation: 空入力チェック・無音チェック・モデル未ロードチェック・推論エラー処理・閾値バリデーションの全動作を不変に保つ_
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2_

  - [x] 3.2 `tests/test_bark_detector.py` のユニットテストを更新する
    - `test_上限超過入力のとき_error_が返ること` を時間長ベースの検証に更新: 10 秒超の入力を生成し、`"Input exceeds maximum duration of 10.0 seconds"` を検証
    - 新規テスト追加: 44.1kHz で 10 秒以内の入力（サンプル数 > 32,000）がエラーなく処理されることの確認
    - 新規テスト追加: ちょうど 10 秒の入力がエラーにならないことの境界値テスト
    - 新規テスト追加: 10 秒をわずかに超える入力がエラーになることの境界値テスト
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 3.3, 3.4_

  - [x] 3.3 `tests/test_bark_detector.py` の Property 3 テストを更新する
    - `test_property_oversized_input_returns_error` のストラテジーを時間長 > 10 秒の入力を生成するように変更
    - ストラテジー: `sample_rate` を `st.integers(8000, 48000)` で生成し、`duration` を `st.floats(10.01, 30.0)` で生成、`len(pcm) = int(sample_rate * duration)`
    - エラーメッセージ検証を `"Input exceeds maximum duration of 10.0 seconds"` に更新
    - `is_bark=False`, `confidence=0.0` の検証を追加
    - _Requirements: 1.2, 2.1, 5.3_

  - [x] 3.4 バグ条件探索テストが PASS することを確認する
    - **Property 1: Expected Behavior** - 10 秒以内の入力は入力長エラーを返さない
    - **IMPORTANT**: タスク 1 で作成した同じテストを再実行する - 新しいテストを書かない
    - タスク 1 のテストは期待動作をエンコードしている
    - このテストが PASS すればバグが修正されたことが確認される
    - 修正後コードでバグ条件探索テストを実行
    - **EXPECTED OUTCOME**: テストが PASS する（バグ修正の確認）
    - _Requirements: 1.1, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 5.1_

  - [x] 3.5 保存プロパティテストが引き続き PASS することを確認する
    - **Property 2: Preservation** - 非バグ条件の入力は修正前後で同一動作
    - **IMPORTANT**: タスク 2 で作成した同じテストを再実行する - 新しいテストを書かない
    - 修正後コードで保存プロパティテストを実行
    - **EXPECTED OUTCOME**: テストが PASS する（リグレッションなしの確認）
    - 全テストが修正後も通過することを確認（動作不変性の保証）
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.2_

  - [x] 3.6 既存スペックドキュメントを更新する
    - `.kiro/specs/bark-check/requirements.md` の Requirement 2.8 を時間長ベースの記述に更新: `IF PcmBlock の時間長 (len(pcm) / sample_rate) が 10.0 秒を超える場合, THEN THE BarkDetector SHALL ...`
    - `.kiro/specs/bark-check/design.md` のエラーハンドリング表を更新: サンプル数ベース → 時間長ベース
    - `.kiro/specs/bark-check/design.md` のモデル仕様・Correctness Properties の関連箇所を更新
    - _Requirements: 2.1, 2.2_

- [x] 4. チェックポイント - 全テストの通過を確認する
  - `pytest tests/test_bark_detector.py` を実行し、全テストが PASS することを確認する
  - 既存のプロパティベーステスト（Property 1〜7）が全て PASS することを確認する
  - 新規追加したバグ条件探索テスト・保存プロパティテストが PASS することを確認する
  - ユニットテスト・プロパティベーステスト合わせて回帰なしを確認する
  - 疑問がある場合はユーザーに確認する
