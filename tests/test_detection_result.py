"""DetectionResult のユニットテストおよびプロパティベーステスト。"""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bark_check.models import DetectionResult


# ---- ユニットテスト ----


class TestDetectionResultToJson:
    """to_json() メソッドのテスト。"""

    def test_正常時に全フィールドが含まれる(self):
        """正常な DetectionResult を JSON にシリアライズしたとき全フィールドが含まれること。"""
        result = DetectionResult(
            is_bark=True, confidence=0.87, timestamp=1718000000.123456,
            audio_duration=1.5, error=None
        )
        data = json.loads(result.to_json())
        assert data["is_bark"] is True
        assert abs(data["confidence"] - 0.87) < 1e-6
        assert abs(data["timestamp"] - 1718000000.123456) < 1.0
        assert abs(data["audio_duration"] - 1.5) < 1e-6
        assert data["error"] is None

    def test_エラーありのとき_error_フィールドが含まれる(self):
        """error フィールドありの DetectionResult をシリアライズしたとき error が含まれること。"""
        result = DetectionResult(
            is_bark=False, confidence=0.0, timestamp=0.0,
            audio_duration=0.0, error="Input PCM block is empty"
        )
        data = json.loads(result.to_json())
        assert data["error"] == "Input PCM block is empty"
        assert data["is_bark"] is False

    def test_is_bark_false_のとき正しくシリアライズされる(self):
        result = DetectionResult(
            is_bark=False, confidence=0.3, timestamp=100.0, audio_duration=2.0
        )
        data = json.loads(result.to_json())
        assert data["is_bark"] is False


class TestDetectionResultFromJson:
    """from_json() クラスメソッドのテスト。"""

    def test_正常な_JSON_から復元できる(self):
        """正常な JSON 文字列から DetectionResult を復元できること。"""
        json_str = json.dumps({
            "is_bark": True,
            "confidence": 0.75,
            "timestamp": 1718000000.0,
            "audio_duration": 1.0,
            "error": None,
        })
        result = DetectionResult.from_json(json_str)
        assert result.is_bark is True
        assert abs(result.confidence - 0.75) < 1e-6
        assert result.error is None

    def test_不正な_JSON_は_error_フィールド付きで返る(self):
        """不正な JSON 文字列が渡されたとき error フィールドを持つ DetectionResult を返すこと。"""
        result = DetectionResult.from_json("{invalid json}")
        assert result.error is not None
        assert "JSON parse error" in result.error

    def test_不正な_JSON_でも例外が発生しない(self):
        """不正な JSON でも from_json() は例外を発生させないこと。"""
        try:
            result = DetectionResult.from_json("not json at all!!!")
        except Exception as e:
            pytest.fail(f"例外が発生した: {e}")

    def test_is_bark_欠落時に_error_フィールドに欠落フィールド名が含まれる(self):
        """is_bark が欠落した JSON のとき error に 'is_bark' が含まれること。"""
        json_str = json.dumps({"confidence": 0.5, "timestamp": 0.0, "audio_duration": 0.0})
        result = DetectionResult.from_json(json_str)
        assert result.error is not None
        assert "is_bark" in result.error

    def test_confidence_欠落時に_error_フィールドに欠落フィールド名が含まれる(self):
        """confidence が欠落した JSON のとき error に 'confidence' が含まれること。"""
        json_str = json.dumps({"is_bark": True, "timestamp": 0.0, "audio_duration": 0.0})
        result = DetectionResult.from_json(json_str)
        assert result.error is not None
        assert "confidence" in result.error

    def test_両必須フィールド欠落時に両フィールド名が_error_に含まれる(self):
        """is_bark と confidence 両方が欠落したとき error に両フィールド名が含まれること。"""
        json_str = json.dumps({"timestamp": 0.0, "audio_duration": 0.0})
        result = DetectionResult.from_json(json_str)
        assert result.error is not None
        assert "is_bark" in result.error
        assert "confidence" in result.error

    def test_timestamp_と_audio_duration_が省略可能(self):
        """timestamp と audio_duration が省略されても from_json() が正常に動作すること。"""
        json_str = json.dumps({"is_bark": False, "confidence": 0.1})
        result = DetectionResult.from_json(json_str)
        assert result.error is None
        assert result.is_bark is False
        assert abs(result.confidence - 0.1) < 1e-6


# ---- プロパティベーステスト ----

# Feature: bark-check, Property 8: DetectionResult のシリアライズ・デシリアライズ ラウンドトリップ
@given(
    is_bark=st.booleans(),
    confidence=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    timestamp=st.floats(0.0, 2e9, allow_nan=False, allow_infinity=False),
    audio_duration=st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property_round_trip(is_bark: bool, confidence: float, timestamp: float, audio_duration: float):
    """Validates: Requirements 6.1, 6.2, 6.3

    任意の有効な DetectionResult に対して to_json() → from_json() でラウンドトリップが成立すること。
    """
    original = DetectionResult(
        is_bark=is_bark,
        confidence=confidence,
        timestamp=timestamp,
        audio_duration=audio_duration,
    )
    restored = DetectionResult.from_json(original.to_json())
    assert restored.is_bark == original.is_bark
    assert abs(restored.confidence - original.confidence) < 1e-6
    assert abs(restored.timestamp - original.timestamp) < 1.0
    assert restored.error is None


# Feature: bark-check, Property 9: 不正 JSON は error フィールドを持つ DetectionResult を返す
@given(
    json_str=st.text().filter(lambda s: _is_invalid_json(s))
)
@settings(max_examples=100)
def test_property_invalid_json_returns_error_result(json_str: str):
    """Validates: Requirements 6.4

    有効な JSON 構文でない任意の文字列に対して from_json() は
    null でない error フィールドを持つ DetectionResult を返すこと。
    """
    result = DetectionResult.from_json(json_str)
    assert result.error is not None
    assert isinstance(result.error, str)
    assert len(result.error) > 0


# Feature: bark-check, Property 10: 必須フィールド欠落 JSON は欠落フィールド名をエラーに含む DetectionResult を返す
@given(
    missing=st.sampled_from([
        ["is_bark"],
        ["confidence"],
        ["is_bark", "confidence"],
    ]),
    extra=st.fixed_dictionaries({
        "timestamp": st.floats(0.0, 1e9, allow_nan=False, allow_infinity=False),
        "audio_duration": st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False),
    }),
)
@settings(max_examples=100)
def test_property_missing_required_fields_error_contains_field_name(
    missing: list[str], extra: dict
):
    """Validates: Requirements 6.5

    必須フィールドが欠落した JSON に対して from_json() は
    error フィールドに欠落フィールド名を含む DetectionResult を返すこと。
    """
    # 全フィールドから欠落分を除いたオブジェクトを作成
    base = {"is_bark": True, "confidence": 0.5, **extra}
    for field in missing:
        base.pop(field, None)
    json_str = json.dumps(base)

    result = DetectionResult.from_json(json_str)
    assert result.error is not None
    for field in missing:
        assert field in result.error


def _is_invalid_json(s: str) -> bool:
    """文字列が不正な JSON かどうかを判定するヘルパー。"""
    try:
        json.loads(s)
        return False
    except (json.JSONDecodeError, ValueError):
        return True
