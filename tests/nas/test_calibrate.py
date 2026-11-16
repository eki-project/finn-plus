"""Tests of the activation-scale calibration and its effect on exported candidates."""

from __future__ import annotations

import pytest

from pathlib import Path

from finn.nas.space import genotype as G
from finn.nas.space import seed_genotypes

torch = pytest.importorskip("torch")
pytest.importorskip("brevitas")
build = pytest.importorskip("finn.nas.space.build")
calibrate_mod = pytest.importorskip("finn.nas.calibrate")
calibrate, synthetic_inputs = calibrate_mod.calibrate, calibrate_mod.synthetic_inputs


def act_scales(net: torch.nn.Module) -> dict[str, float]:
    """Return the scale of every activation quantizer (input quantizer and joins included)."""
    return {
        name: float(module.act_quant.scale())
        for name, module in net.named_modules()
        if hasattr(module, "act_quant") and hasattr(module.act_quant, "scale")
    }


@pytest.mark.util
class TestCalibrate:
    """Calibration turns the degenerate initial scales into data-dependent ones."""

    def test_initial_scales_are_degenerate(self) -> None:
        net = build.to_module(seed_genotypes()["residual"]).eval()
        scales = act_scales(net)
        # 4-bit activations: 1 / 15; the 5-bit signed join: 1 / 16 (int_max of a 5-bit signed
        # narrow-range... no: non-narrow -> 15); the input quantizer is a constant 2**-7.
        assert scales["input_quant"] == pytest.approx(2.0**-7)
        acts = [v for k, v in scales.items() if k != "input_quant"]
        assert len(set(round(v, 6) for v in acts)) <= 2
        assert all(v < 0.1 for v in acts)

    def test_calibrated_scales_follow_the_data(self) -> None:
        torch.manual_seed(0)
        net = build.to_module(seed_genotypes()["residual"])
        inputs = synthetic_inputs((3, 32, 32), 64, seed=0)
        calibrate(net, inputs, steps=5, batch_size=32)
        assert not net.training
        scales = act_scales(net)
        acts = {k: v for k, v in scales.items() if k != "input_quant"}
        assert all(0.1 < v < 10.0 for v in acts.values()), acts
        assert len(set(round(v, 6) for v in acts.values())) == len(acts)
        # the same data gives the same scales
        torch.manual_seed(0)
        again = build.to_module(seed_genotypes()["residual"])
        calibrate(again, inputs, steps=5, batch_size=32)
        assert act_scales(again) == pytest.approx(scales)

    def test_zero_steps_only_switches_to_eval(self) -> None:
        net = build.to_module(seed_genotypes()["mlp_like"])
        before = act_scales(net.eval())
        calibrate(net.train(), synthetic_inputs((3, 32, 32), 4), steps=0)
        assert not net.training and act_scales(net) == before

    def test_too_few_inputs(self) -> None:
        net = build.to_module(seed_genotypes()["mlp_like"])
        with pytest.raises(ValueError):
            calibrate(net, synthetic_inputs((3, 32, 32), 1))

    def test_join_operands_share_one_scale(self, tmp_path: Path) -> None:
        """Both operands of a join are quantized by the same module, which exports as two
        Quant nodes with the same scale initializer feeding the Add."""
        import onnx
        from onnx import numpy_helper

        from finn.nas.export import export_candidate

        torch.manual_seed(0)
        net = build.to_module(seed_genotypes()["residual"])
        calibrate(net, synthetic_inputs((3, 32, 32), 32), steps=3, batch_size=16)
        path = export_candidate(net, (1, 3, 32, 32), tmp_path / "residual.onnx")
        model = onnx.load(str(path))
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        producers = {o: n for n in model.graph.node for o in n.output}
        (add,) = [n for n in model.graph.node if n.op_type == "Add"]
        operand_scales = []
        for name in add.input:
            quant = producers[name]
            assert quant.op_type == "Quant"
            operand_scales.append(float(inits[quant.input[1]]))
        assert operand_scales[0] == operand_scales[1]
        assert float(inits[producers[add.input[0]].input[3]]) == 5  # join_bits of the seed
