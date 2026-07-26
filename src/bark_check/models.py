"""判定結果データモデルモジュール。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class DetectionResult:
    """判定結果を表すデータ構造。

    吠え声の有無、信頼度スコア、タイムスタンプ、音声長、エラー情報を保持する。
    """

    is_bark: bool
    confidence: float
    timestamp: float
    audio_duration: float
    error: str | None = None

    def to_json(self) -> str:
        """DetectionResult を JSON 文字列にシリアライズする。

        Returns:
            is_bark, confidence, timestamp, audio_duration, error を含む JSON 文字列。
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "DetectionResult":
        """JSON 文字列から DetectionResult をデシリアライズする。

        パースエラーや必須フィールド欠落の場合は、error フィールドにメッセージを
        格納した DetectionResult を返す。例外は発生しない。

        Args:
            json_str: デシリアライズ対象の JSON 文字列。

        Returns:
            復元された DetectionResult。エラー時は error フィールドにメッセージを含む。
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return cls(
                is_bark=False,
                confidence=0.0,
                timestamp=0.0,
                audio_duration=0.0,
                error=f"JSON parse error: {e}",
            )

        missing = [f for f in ("is_bark", "confidence") if f not in data]
        if missing:
            return cls(
                is_bark=False,
                confidence=0.0,
                timestamp=0.0,
                audio_duration=0.0,
                error=f"Missing required fields: {', '.join(missing)}",
            )

        return cls(
            is_bark=bool(data["is_bark"]),
            confidence=float(data["confidence"]),
            timestamp=float(data.get("timestamp", 0.0)),
            audio_duration=float(data.get("audio_duration", 0.0)),
            error=data.get("error"),
        )
