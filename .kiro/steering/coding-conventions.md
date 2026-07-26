# コーディング規約

## Python docstrings

すべての Python コードの docstring は **日本語 Google スタイル** で記述する。

### フォーマット

```python
def example(arg1: int, arg2: str) -> bool:
    """関数の概要を1行で記述する。

    必要に応じて詳細説明を続ける。
    複数行になってもよい。

    Args:
        arg1: 引数の説明。
        arg2: 引数の説明。

    Returns:
        戻り値の説明。

    Raises:
        ValueError: 発生条件の説明。
        TypeError: 発生条件の説明。
    """
```

### ルール

- 概要行は動詞で始める（例: 「音声ファイルを読み込む。」「吠え声を検出する。」）
- セクション見出し（Args:, Returns:, Raises:）は英語のまま維持する（Google スタイルの慣習）
- 各引数・戻り値・例外の説明は日本語で記述する
- クラスの docstring は概要のみでよい（`__init__` の引数は `__init__` の docstring に記述する）
- 1行で収まる場合も三重引用符を使用する
