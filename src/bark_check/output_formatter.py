"""判定結果のフォーマットモジュール。"""

from bark_check.models import DetectionResult


class OutputFormatter:
    """判定結果のフォーマットを担当する CLI レイヤーのモジュール。"""

    def format_text(self, result: DetectionResult) -> str:
        """判定結果をテキスト形式でフォーマットする。

        Args:
            result: フォーマット対象の判定結果。

        Returns:
            吠え声の有無と信頼度を含むテキスト文字列。
        """
        if result.is_bark:
            return f"Bark detected (confidence: {result.confidence:.2f})"
        return f"No bark detected (confidence: {result.confidence:.2f})"

    def format_json(self, result: DetectionResult) -> str:
        """判定結果を JSON 形式でフォーマットする。

        Args:
            result: フォーマット対象の判定結果。

        Returns:
            DetectionResult の JSON 文字列。
        """
        return result.to_json()
