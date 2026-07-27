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
