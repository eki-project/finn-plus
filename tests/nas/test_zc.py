"""Unit tests for the zero-cost proxies (finn.nas.zc) on float and Brevitas networks."""

from __future__ import annotations

import pytest

import json
import math
import torch
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant.binary import SignedBinaryActPerTensorConst, SignedBinaryWeightPerTensorConst
from torch import nn

import finn.nas.zc.measures as zc_measures
from finn.nas.zc import DEFAULT_MEASURES, ProxyResult, available_measures, compute_proxies
from finn.nas.zc.measures.brevitas_compat import activation_modules

NUM_CLASSES = 4


class FloatCNN(nn.Module):
    """Two float conv blocks and a classifier."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, bias=True)
        self.bn2 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 3 * 3, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        return self.fc(x.flatten(1))


class QuantCNN(nn.Module):
    """Brevitas counterpart of FloatCNN (pattern of tests/nas/models.py TinyCNN)."""

    def __init__(self, channels: int = 4, wbits: int = 4, abits: int = 4) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=8)
        self.conv1 = QuantConv2d(3, channels, 3, padding=1, bias=False, weight_bit_width=wbits)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = QuantReLU(bit_width=abits)
        self.conv2 = QuantConv2d(channels, 2 * channels, 3, bias=False, weight_bit_width=wbits)
        self.bn2 = nn.BatchNorm2d(2 * channels)
        self.act2 = QuantReLU(bit_width=abits)
        self.pool = nn.MaxPool2d(2)
        self.fc = QuantLinear(2 * channels * 3 * 3, NUM_CLASSES, bias=False, weight_bit_width=wbits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        x = self.act1(self.bn1(self.conv1(self.inp(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        return self.fc(x.flatten(1))


class QuantResCNN(nn.Module):
    """Wider Brevitas network with a residual add and a quantized bias."""

    def __init__(self, channels: int = 12, wbits: int = 2, abits: int = 2) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=8)
        self.stem = QuantConv2d(3, channels, 3, padding=1, bias=False, weight_bit_width=wbits)
        self.bn0 = nn.BatchNorm2d(channels)
        self.act0 = QuantReLU(bit_width=abits)
        self.conv1 = QuantConv2d(
            channels, channels, 3, padding=1, bias=False, weight_bit_width=wbits
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = QuantReLU(bit_width=abits)
        self.conv2 = QuantConv2d(
            channels, channels, 3, padding=1, bias=False, weight_bit_width=wbits
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = QuantReLU(bit_width=abits)
        self.pool = nn.AvgPool2d(2)
        self.fc = QuantLinear(channels * 4 * 4, NUM_CLASSES, bias=True, weight_bit_width=wbits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        x = self.act0(self.bn0(self.stem(self.inp(x))))
        y = self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))
        x = self.act2(x + y)
        return self.fc(self.pool(x).flatten(1))


class BipolarCNN(nn.Module):
    """1-bit weights and activations. Its weight quantizer clamps the latent weight in place
    on every forward, so it exposes any measure that runs on the caller's model."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=8)
        self.conv = QuantConv2d(3, 8, 3, bias=False, weight_quant=SignedBinaryWeightPerTensorConst)
        self.bn = nn.BatchNorm2d(8)
        self.act = QuantIdentity(act_quant=SignedBinaryActPerTensorConst)
        self.pool = nn.MaxPool2d(2)
        self.fc = QuantLinear(
            8 * 3 * 3, NUM_CLASSES, bias=False, weight_quant=SignedBinaryWeightPerTensorConst
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        return self.fc(self.pool(self.act(self.bn(self.conv(self.inp(x))))).flatten(1))


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed minibatch of 16 images with 4 samples per class."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(16, 3, 8, 8, generator=generator), torch.arange(16) % NUM_CLASSES


def _build(cls: type[nn.Module]) -> nn.Module:
    """Build a network with reproducible initial weights."""
    torch.manual_seed(0)
    return cls()


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of the state of ``model``."""
    return {name: value.clone() for name, value in model.state_dict().items()}


@pytest.fixture(scope="module")
def brevitas_results() -> dict[str, ProxyResult]:
    """Default measures on the two Brevitas networks, computed once for the module."""
    inputs, targets = _batch()
    return {
        cls.__name__: compute_proxies(_build(cls), inputs, targets, num_classes=NUM_CLASSES)
        for cls in (QuantCNN, QuantResCNN)
    }


@pytest.mark.util
class TestZeroCostProxies:
    """Public API of finn.nas.zc."""

    def test_measure_lists(self) -> None:
        assert set(DEFAULT_MEASURES) <= set(available_measures())
        assert {"params", "flops"} <= set(DEFAULT_MEASURES)
        # available, but not meaningful on Brevitas networks
        assert {"zen", "synflow_bn"} <= set(available_measures()) - set(DEFAULT_MEASURES)

    @pytest.mark.parametrize("measure", DEFAULT_MEASURES)
    def test_brevitas_finite(self, brevitas_results: dict[str, ProxyResult], measure: str) -> None:
        for result in brevitas_results.values():
            assert result.errors == {}
            assert math.isfinite(result.scores[measure])
            assert result.seconds[measure] >= 0.0

    @pytest.mark.parametrize("measure", DEFAULT_MEASURES)
    def test_brevitas_architecture_dependent(
        self, brevitas_results: dict[str, ProxyResult], measure: str
    ) -> None:
        plain, residual = (brevitas_results[name].scores[measure] for name in brevitas_results)
        assert plain != 0.0
        assert not math.isclose(plain, residual, rel_tol=1e-3)

    def test_float_network(self) -> None:
        inputs, targets = _batch()
        result = compute_proxies(
            _build(FloatCNN), inputs, targets, available_measures(), num_classes=NUM_CLASSES
        )
        assert result.errors == {}
        assert all(math.isfinite(result.scores[name]) for name in available_measures())

    @pytest.mark.parametrize("cls", [FloatCNN, QuantCNN, QuantResCNN, BipolarCNN])
    @pytest.mark.parametrize("training", [True, False])
    def test_model_untouched(self, cls: type[nn.Module], training: bool) -> None:
        inputs, targets = _batch()
        model = _build(cls).train(training)
        before = _state(model)
        rng_state = torch.get_rng_state()
        result = compute_proxies(
            model, inputs, targets, available_measures(), num_classes=NUM_CLASSES
        )
        after = model.state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert all(module.training is training for module in model.modules())
        assert all(param.grad is None for param in model.parameters())
        assert not inputs.requires_grad
        assert torch.equal(rng_state, torch.get_rng_state())
        assert all(math.isfinite(result.scores[name]) for name in DEFAULT_MEASURES)

    def test_training_flag_does_not_change_scores(self) -> None:
        inputs, targets = _batch()
        model = _build(QuantCNN)
        train = compute_proxies(model.train(), inputs, targets, num_classes=NUM_CLASSES)
        evaluation = compute_proxies(model.eval(), inputs, targets, num_classes=NUM_CLASSES)
        assert train.scores == evaluation.scores

    @pytest.mark.parametrize("cls", [QuantCNN, QuantResCNN])
    def test_deterministic(self, cls: type[nn.Module]) -> None:
        inputs, targets = _batch()
        model = _build(cls)
        first = compute_proxies(
            model, inputs, targets, available_measures(), num_classes=NUM_CLASSES
        )
        torch.rand(3)  # the caller's random stream must not matter
        second = compute_proxies(
            model, inputs, targets, available_measures(), num_classes=NUM_CLASSES
        )
        assert first.scores == second.scores
        assert first.errors == second.errors == {}

    def test_seed_selects_random_inputs(self) -> None:
        # zen is the only measure that draws random numbers
        inputs, targets = _batch()
        model = _build(FloatCNN)
        scores = [
            compute_proxies(model, inputs, targets, ["zen"], num_classes=NUM_CLASSES, seed=seed)
            for seed in (0, 0, 1)
        ]
        assert scores[0].scores == scores[1].scores
        assert scores[0].scores != scores[2].scores

    def test_failure_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs, targets = _batch()
        model = _build(QuantCNN)
        reference = compute_proxies(model, inputs, targets, num_classes=NUM_CLASSES)

        def broken(*_args: object, **_kwargs: object) -> None:
            """Stand-in for a measure that raises."""
            raise RuntimeError("proxy exploded")

        monkeypatch.setitem(zc_measures._measure_impls, "snip", broken)
        result = compute_proxies(model, inputs, targets, num_classes=NUM_CLASSES)
        assert math.isnan(result.scores["snip"])
        assert result.errors == {"snip": "RuntimeError: proxy exploded"}
        assert set(result.scores) == set(DEFAULT_MEASURES) == set(result.seconds)
        for name in set(DEFAULT_MEASURES) - {"snip"}:
            assert result.scores[name] == reference.scores[name]

    def test_non_finite_score_is_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs, targets = _batch()
        monkeypatch.setitem(zc_measures._measure_impls, "nwot", lambda *_a, **_k: -math.inf)
        result = compute_proxies(
            _build(QuantCNN), inputs, targets, ["nwot", "params"], num_classes=NUM_CLASSES
        )
        assert math.isnan(result.scores["nwot"])
        assert "non-finite" in result.errors["nwot"]
        assert math.isfinite(result.scores["params"])

    @pytest.mark.parametrize("cls", [FloatCNN, QuantCNN, QuantResCNN])
    def test_params(self, cls: type[nn.Module]) -> None:
        inputs, targets = _batch()
        model = _build(cls)
        result = compute_proxies(model, inputs, targets, ["params"], num_classes=NUM_CLASSES)
        assert result.scores["params"] == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    @pytest.mark.parametrize("cls", [FloatCNN, QuantCNN])
    def test_flops(self, cls: type[nn.Module]) -> None:
        # multiply-accumulates per sample: conv1 8x8x4 outputs x 27, conv2 6x6x8 outputs x 36,
        # fc 4 outputs x 72
        inputs, targets = _batch()
        result = compute_proxies(_build(cls), inputs, targets, ["flops"], num_classes=NUM_CLASSES)
        assert result.scores["flops"] == 8 * 8 * 4 * 27 + 6 * 6 * 8 * 36 + 4 * 72

    def test_bipolar_synflow_counts_all_layers(self) -> None:
        # With the quantized forward the bipolar activation would block the gradient of the
        # first layer, and the score would be that of the classifier alone.
        inputs, targets = _batch()
        value = zc_measures.calc_measure(
            "synflow", _build(BipolarCNN), torch.device("cpu"), inputs, targets
        )
        assert len(value) == 2
        assert all(float(layer.detach().sum()) > 0.0 for layer in value)

    def test_nwot_counts_each_activation_once(self) -> None:
        # Same topology and sign patterns: a QuantReLU wraps a float ReLU, which must not be
        # counted a second time, and the input quantizer must not be counted at all.
        inputs, targets = _batch()
        quant, floating = _build(QuantCNN), _build(FloatCNN)
        assert [type(m).__name__ for m in activation_modules(quant)] == [
            "QuantIdentity",
            "QuantReLU",
            "QuantReLU",
        ]
        assert [type(m).__name__ for m in activation_modules(floating)] == ["ReLU"]
        scores = [
            compute_proxies(model, inputs, targets, ["nwot"], num_classes=NUM_CLASSES).scores
            for model in (quant, floating)
        ]
        assert abs(scores[0]["nwot"] - scores[1]["nwot"]) < 0.5 * len(targets) * math.log(2)

    def test_result_is_json_serializable(self, brevitas_results: dict[str, ProxyResult]) -> None:
        result = brevitas_results["QuantCNN"]
        restored = json.loads(json.dumps(result.to_dict()))
        assert restored == {"scores": result.scores, "errors": {}, "seconds": result.seconds}

    def test_invalid_arguments(self) -> None:
        inputs, targets = _batch()
        model = _build(QuantCNN)
        with pytest.raises(ValueError, match="Unknown measures"):
            compute_proxies(model, inputs, targets, ["no_such_proxy"], num_classes=NUM_CLASSES)
        with pytest.raises(ValueError, match="Targets span"):
            compute_proxies(model, inputs, targets, ["params"], num_classes=NUM_CLASSES - 1)
