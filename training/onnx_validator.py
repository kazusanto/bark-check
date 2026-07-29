"""ONNX モデルの CoreML 互換性を検証するバリデータ。"""

import onnx

EXPECTED_INPUT_SHAPE = [1, 1, 40, 199]
EXPECTED_OUTPUT_SHAPE = [1, 1]
EXPECTED_OPSET_VERSION = 9
ALLOWED_OPERATORS = frozenset(
    {
        "Conv",
        "Relu",
        "BatchNormalization",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Reshape",
        "Gemm",
        "Sigmoid",
        "Constant",
        "Squeeze",
        "Transpose",  # Conv2d モデルの permute 用
    }
)


def validate_onnx_for_coreml(model_path: str) -> list[str]:
    """ONNX モデルの CoreML 互換性を検証する。

    検証項目:
    1. 入力 shape が [1, 40, 199] であること
    2. 出力 shape が [1, 1] であること
    3. opset_version が 9 であること
    4. 全次元が固定整数値（dim_value）であること（動的軸なし）
    5. 使用オペレータが許可リスト内であること

    許可オペレータ: Conv, Relu, BatchNormalization, MaxPool,
    AveragePool, GlobalAveragePool, Reshape, Gemm, Sigmoid, Constant

    Args:
        model_path: 検証対象の ONNX ファイルパス。

    Returns:
        失敗した検証項目のエラーメッセージリスト。空リストなら全項目合格。
    """
    errors: list[str] = []
    model = onnx.load(model_path)

    # 1. 入力 shape の検証
    input_shape = _get_tensor_shape(model.graph.input[0])
    if input_shape != EXPECTED_INPUT_SHAPE:
        errors.append(
            f"Input shape mismatch: expected {EXPECTED_INPUT_SHAPE}, got {input_shape}"
        )

    # 2. 出力 shape の検証
    output_shape = _get_tensor_shape(model.graph.output[0])
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        errors.append(
            f"Output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, got {output_shape}"
        )

    # 3. opset_version の検証
    opset_version = model.opset_import[0].version
    if opset_version != EXPECTED_OPSET_VERSION:
        errors.append(
            f"Opset version mismatch: expected {EXPECTED_OPSET_VERSION}, got {opset_version}"
        )

    # 4. 動的軸の検証（入力・出力の全次元が固定整数値であること）
    errors.extend(_check_dynamic_axes(model.graph.input, "input"))
    errors.extend(_check_dynamic_axes(model.graph.output, "output"))

    # 5. 使用オペレータの検証
    errors.extend(_check_operators(model.graph.node))

    return errors


def _get_tensor_shape(tensor_info: onnx.TensorProto) -> list[int]:
    """テンソル情報から shape を整数リストとして取得する。

    Args:
        tensor_info: ONNX テンソル情報。

    Returns:
        各次元の値をリストとして返す。動的軸は 0 として返す。
    """
    dims = tensor_info.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.dim_value > 0:
            shape.append(d.dim_value)
        else:
            shape.append(0)
    return shape


def _check_dynamic_axes(
    tensors: list, tensor_type: str
) -> list[str]:
    """テンソルの全次元が固定整数値であることを検証する。

    Args:
        tensors: 検証対象のテンソル情報リスト。
        tensor_type: テンソルの種類を示す文字列（"input" または "output"）。

    Returns:
        動的軸が見つかった場合のエラーメッセージリスト。
    """
    errors: list[str] = []
    for tensor_info in tensors:
        dims = tensor_info.type.tensor_type.shape.dim
        for i, d in enumerate(dims):
            if d.dim_param:
                errors.append(
                    f"Dynamic axis found: {tensor_type} dim {i} has symbolic name '{d.dim_param}'"
                )
            elif d.dim_value <= 0:
                errors.append(
                    f"Dynamic axis found: {tensor_type} dim {i} has no fixed value"
                )
    return errors


def _check_operators(nodes: list) -> list[str]:
    """使用オペレータが許可リスト内であることを検証する。

    Args:
        nodes: ONNX グラフのノードリスト。

    Returns:
        未許可オペレータが見つかった場合のエラーメッセージリスト。
    """
    errors: list[str] = []
    seen_unsupported: set[str] = set()
    allowed_str = ", ".join(sorted(ALLOWED_OPERATORS))

    for node in nodes:
        if node.op_type not in ALLOWED_OPERATORS and node.op_type not in seen_unsupported:
            seen_unsupported.add(node.op_type)
            errors.append(
                f"Unsupported operator: {node.op_type} (allowed: {allowed_str})"
            )

    return errors
