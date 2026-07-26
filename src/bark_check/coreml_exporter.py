"""CoreML エクスポートモジュール。"""

from __future__ import annotations

import os


class CoreMLExporter:
    """ONNX モデルを CoreML 形式 (.mlmodel) にエクスポートする。"""

    def export(self, model_path: str, output_path: str) -> str:
        """ONNX モデルを CoreML 形式にエクスポートする。

        Args:
            model_path: 入力 ONNX モデルファイルのパス。
            output_path: 出力 .mlmodel ファイルのパス。

        Returns:
            エクスポートされたファイルのパス。

        Raises:
            FileNotFoundError: model_path が存在しない場合。
            RuntimeError: 変換に失敗した場合。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            import coremltools as ct

            model = ct.convert(
                model_path,
                minimum_deployment_target=ct.target.iOS13,
                compute_units=ct.ComputeUnit.ALL,
            )

            # 入力名・出力名を設定
            spec = model.get_spec()

            # 入力名を pcm_features に変更
            for inp in spec.description.input:
                inp.name = "pcm_features"

            # 出力名を bark_probability に変更
            for out in spec.description.output:
                out.name = "bark_probability"

            # 更新した spec からモデルを再構築
            model = ct.models.MLModel(spec)

            model.save(output_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"CoreML conversion failed: {e}") from e

        return output_path
