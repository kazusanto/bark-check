# 言語規則

## 仕様書・設計書の記述言語

このプロジェクトでは、仕様・設計ドキュメントの説明文・要件・コメントは原則 **日本語** で記述する。
ただし、英語が自然または必要な箇所（下記「例外」参照）は英語を優先する。

対象ファイル:
- `requirements.md`（要件定義）
- `design.md`（設計ドキュメント）
- `tasks.md`（実装タスクリスト）

### 適用範囲

以下の要素は日本語で記述すること:

- Introduction / Overview などの説明文
- 用語集（Glossary）
- User Story および Acceptance Criteria
- Correctness Properties の説明文（プロパティタイトルと本文）
- エラーハンドリングの説明
- テスト戦略の説明

### 例外

以下は英語・原文のまま維持する:

- コードブロック内のコード・コメント（Python, JSON 等）
- EARS パターンのキーワード（SHALL, WHEN, IF, THEN, WHERE 等）
- 固有名詞・技術用語（BarkDetector, DetectionResult, MFCC, CoreML 等）
- Mermaid ダイアグラムのノードラベル
- テーブル内のコード・型名（`bool`, `float`, `str` 等）
- Kiro spec の規定フォーマットが定める固定表記・見出し（文書構造やチェッカーの動作に影響するもの）
- Git コミットメッセージの subject line（1行目）は英語で記述する。body（2行目以降）は日本語でもよい

## EARS 構文の記述規則

User Story および Acceptance Criteria は **Kiro デフォルト構文に準拠した日本語** で記述する。

### User Story の書き方

Kiro の標準 User Story フォーマット（As a / I want / so that）を使用する。キーワードは英語のまま、本文は日本語で記述する:

```
As a <ロール>, I want <欲しい機能や振る舞い>, so that <理由・目的>
```

例:
- `As a 開発者, I want PcmBlock を渡すだけで犬の吠え声かどうかを判定したい, so that 手動でラベル付けする手間を省ける`
- `As a iOS アプリ開発者, I want CoreML で正しく認識される 4D 入力形状のモデルが欲しい, so that Xcode でのモデル統合がスムーズに行える`

### Acceptance Criteria の書き方

EARS パターンのキーワードを混ぜた日本語で記述する:

| パターン | 構文例 |
|---|---|
| Ubiquitous | `THE <主語> SHALL <動作>。` |
| Event-driven | `WHEN <条件>, THE <主語> SHALL <動作>。` |
| State-driven | `WHILE <状態>, THE <主語> SHALL <動作>。` |
| Conditional | `IF <条件>, THEN THE <主語> SHALL <動作>。` |
| Optional | `WHERE <機能が有効>, THE <主語> SHALL <動作>。` |
| Quantified | `FOR ALL <対象>, THE <主語> SHALL <動作>。` |

例:
- `THE BarkCNN2d SHALL Conv2d レイヤーを kernel_size (3, 3) で使用し、pooling 前の空間次元を維持する`
- `WHEN 抽出した MFCC フレーム数が 199 未満の場合, THE ESC50BarkDataset SHALL 時間軸に沿ってゼロパディングする`
- `IF PcmBlock が空（サンプル数ゼロ）である場合, THEN THE BarkDetector SHALL 空入力エラーを示すエラー種別を含む DetectionResult を返す`

## 時系列依存表現の禁止

main-spec（requirements.md, design.md, tasks.md）およびコード中のコメント・docstring では、開発の時系列を前提とした表現を使用しない。

更新計画を記述する sub-spec ではこの制約は適用しない（変更前後の比較が本質的に必要なため）。

### 禁止する表現の例

| 避けるべき表現 | 代替 |
|---|---|
| 新仕様 / 旧仕様 | 具体的な仕様名やバージョンで参照する |
| 従来の実装 / 新しい実装 | クラス名・関数名で具体的に参照する |
| 以前は〜だった | 現在の仕様のみを記述する |
| 今後〜に変更する | 変更先の仕様をそのまま記述する |

### 理由

- ドキュメントやコメントは時間が経つと「新」「旧」の基準が不明になる
- 読み手がプロジェクトの変更履歴を知らなくても理解できる記述にする
- 変更経緯が必要な場合は Git の履歴やコミットメッセージで追う

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
