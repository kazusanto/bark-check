# 言語規則

## 仕様書・設計書の記述言語

このプロジェクトでは、仕様・設計ドキュメントの説明文・要件・コメントは原則 **日本語** で記述する。
ただし、英語が自然または必要な箇所（下記「例外」参照）は英語を優先する。

対象ファイル:
- `requirements.md`（要件定義）
- `design.md`（設計ドキュメント）
- `tasks.md`（実装タスクリスト）

## 適用範囲

以下の要素は日本語で記述すること:

- Introduction / Overview などの説明文
- 用語集（Glossary）
- User Story および Acceptance Criteria
- Correctness Properties の説明文（プロパティタイトルと本文）
- エラーハンドリングの説明
- テスト戦略の説明

## 例外

以下は英語・原文のまま維持する:

- コードブロック内のコード・コメント（Python, JSON 等）
- EARS パターンのキーワード（SHALL, WHEN, IF, THEN, WHERE 等）
- 固有名詞・技術用語（BarkDetector, DetectionResult, MFCC, CoreML 等）
- Mermaid ダイアグラムのノードラベル
- テーブル内のコード・型名（`bool`, `float`, `str` 等）
- Kiro spec の規定フォーマットが定める固定表記・見出し（文書構造やチェッカーの動作に影響するもの）
- Git コミットメッセージの subject line（1行目）は英語で記述する。body（2行目以降）は日本語でもよい

## Git コミットメッセージ

[Conventional Commits](https://www.conventionalcommits.org/) に準拠する。

### フォーマット

```
<type>(<scope>): <subject>
```

- `<type>` は必須。以下のいずれかを使用する:
  - `feat` — 新機能
  - `fix` — バグ修正
  - `refactor` — リファクタリング（機能変更なし・バグ修正なし）
  - `docs` — ドキュメントのみの変更
  - `test` — テストの追加・修正
  - `chore` — ビルド・CI・依存関係など雑務
  - `perf` — パフォーマンス改善
  - `style` — フォーマット変更（コードの意味に影響しない）
  - `ci` — CI 設定の変更
- `<scope>` は任意。変更対象のモジュールや領域を示す（例: `training`, `cli`, `detector`）
- `<subject>` は英語・小文字始まり・末尾にピリオドを付けない・命令形で記述する

### 例

```
feat(detector): add confidence threshold parameter
fix(training): lower ONNX opset_version to 9 for CoreML compatibility
docs: update README with installation instructions
refactor(cli): extract output formatting into separate module
```
