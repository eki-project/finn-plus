"""Tests of genotype -> Brevitas module: shapes, parameter counts, quantizers, export."""

from __future__ import annotations

import pytest

import random
from pathlib import Path

from finn.nas.space import SearchSpace, SpaceConfig, count_params
from finn.nas.space import genotype as G
from finn.nas.space import infer_shapes, seed_genotypes
from finn.nas.space.genotype import InvalidGenotypeError, iter_nodes

torch = pytest.importorskip("torch")
pytest.importorskip("brevitas")
build = pytest.importorskip("finn.nas.space.build")
nn = torch.nn
Branching, GenotypeNet, to_module = build.Branching, build.GenotypeNet, build.to_module

N_SAMPLES = 200
BATCH = 2


def module_at(net: GenotypeNet, path: tuple[int, ...]) -> nn.Module:
    """Return the module built for the body node at ``path``; the module tree mirrors it."""
    module = net.body
    for index in path:
        module = module.branches[index] if isinstance(module, Branching) else module[index]
    return module


def real_shapes(net: GenotypeNet, paths: list[tuple[int, ...]], x: torch.Tensor) -> dict:
    """Run ``x`` through ``net`` and record the output shape (without batch) of every node."""
    shapes: dict[tuple[int, ...], tuple[int, ...]] = {}
    handles = []
    for path in paths:

        def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor, path=path) -> None:
            shapes[path] = tuple(output.shape[1:])

        handles.append(module_at(net, path).register_forward_hook(hook))
    try:
        net(x)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def n_weights(net: nn.Module) -> int:
    """Number of parameters without the learned quantizer scales."""
    return sum(p.numel() for p in build.weight_parameters(net))


def warm_up(net: nn.Module, x: torch.Tensor, steps: int = 2) -> nn.Module:
    """Initialize the activation scales with a few train-mode passes; return ``net`` in eval."""
    net.train()
    with torch.no_grad():
        for _ in range(steps):
            net(x)
    return net.eval()


@pytest.mark.util
class TestBuild:
    """Random genotypes build, run and match the analytic shapes and parameter counts."""

    def test_random_samples(self) -> None:
        config = SpaceConfig(max_params=50_000, max_nodes=30)
        space = SearchSpace(config)
        torch.manual_seed(0)
        x = torch.randn(BATCH, *config.input_shape)
        for seed in range(N_SAMPLES):
            genotype = space.sample(random.Random(seed))
            info = infer_shapes(genotype)
            net = to_module(genotype)
            logits = net(x)
            assert logits.shape == (BATCH, config.num_classes)
            assert logits.dtype == torch.float32
            assert torch.isfinite(logits).all()
            assert n_weights(net) == count_params(genotype)
            assert real_shapes(net, list(info.nodes), x) == {
                path: node.out_shape for path, node in info.nodes.items()
            }

    def test_eval_mode_and_single_image(self) -> None:
        space = SearchSpace(SpaceConfig(max_params=50_000))
        for seed in range(20):
            genotype = space.sample(random.Random(seed))
            net = warm_up(to_module(genotype), torch.randn(4, 3, 32, 32))
            with torch.no_grad():
                assert net(torch.randn(1, 3, 32, 32)).shape == (1, 10)

    def test_parameter_definition(self) -> None:
        """Weights of conv/linear, the affine parameters of batch norm and the learned
        activation scales, nothing else."""
        genotype = seed_genotypes()["residual"]
        net = to_module(genotype)
        names = {name.rsplit(".", 1)[-1] for name, _ in net.named_parameters()}
        assert names == {"weight", "bias", "value"}
        assert all(
            isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
            for module in net.modules()
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter)
        )
        scales = [n for n, _ in net.named_parameters() if build.is_scale_parameter(n)]
        # three 4-bit activations and the join quantizer; weights and the input quantizer
        # have no scale parameter (statistics resp. constant)
        assert len(scales) == 4
        assert n_weights(net) == count_params(genotype)

    def test_invalid_genotype_is_rejected(self) -> None:
        with pytest.raises(InvalidGenotypeError):
            to_module(G.make_genotype(G.sequential(G.linear(16))))


@pytest.mark.util
class TestQuantizers:
    """The fixed-point conventions the FINN mapping relies on."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_activations_are_on_a_learned_grid(self, bits: int) -> None:
        act, fmt = build.quant_activation(bits)
        assert fmt.bits == bits and not fmt.signed and fmt.source is act
        x = torch.linspace(-3, 30, 1001)
        warm_up(act, x)
        y = act(x)
        scale = fmt.scale()
        assert 0.0 < float(scale) < 30.0
        assert torch.allclose(y / scale, (y / scale).round(), atol=1e-5)
        assert y.min() == 0 and y.max() == pytest.approx(float(scale) * (2**bits - 1))
        assert len(torch.unique(y)) == 2**bits

    def test_bipolar_activation(self) -> None:
        act, fmt = build.quant_activation(1)
        assert fmt is None
        y = act(torch.tensor([-2.0, -0.1, 0.1, 2.0]))
        assert set(y.tolist()) == {-1.0, 1.0}

    @pytest.mark.parametrize("bits", [2, 5])
    def test_join_quantizer_is_signed(self, bits: int) -> None:
        quant, fmt = build.join_quantizer(bits)
        assert fmt.bits == bits and fmt.signed and fmt.source is quant
        x = torch.linspace(-20, 20, 1001)
        warm_up(quant, x)
        y = quant(x)
        scale = fmt.scale()
        assert torch.allclose(y / scale, (y / scale).round(), atol=1e-5)
        assert y.min() < 0 < y.max()
        assert len(torch.unique(y)) <= 2**bits

    @pytest.mark.parametrize("bits", [2, 8])
    def test_input_quantizer(self, bits: int) -> None:
        quantizer, fmt = build.input_quantizer(bits)
        step = 2.0 ** (1 - bits)
        assert (fmt.bits, fmt.signed, float(fmt.scale())) == (bits, True, step)
        y = quantizer(torch.linspace(-2, 2, 4001))
        assert y.min() == -1.0 and y.max() == 1.0 - step
        assert torch.allclose(y / step, (y / step).round())
        assert len(torch.unique(y)) == 2**bits

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_weights(self, bits: int) -> None:
        """Narrow-range integers at a per-tensor scale that follows the largest weight."""
        torch.manual_seed(0)
        conv = build._conv(G.conv(8, 3, weight_bits=bits), (3, 8, 8))
        q = conv.quant_weight()
        values = q.value.detach()
        if bits == 1:
            assert set(values.unique().tolist()) == {-1.0, 1.0}
            return
        scale = float(q.scale)
        levels = 2 ** (bits - 1) - 1
        assert scale == pytest.approx(float(conv.weight.abs().max()) / levels)
        assert torch.allclose(values / scale, (values / scale).round(), atol=1e-5)
        assert values.abs().max() == pytest.approx(scale * levels)
        levels_used = len(torch.unique(values))
        if bits <= 4:
            assert levels_used == 2**bits - 1
        else:
            assert 2**bits // 2 < levels_used <= 2**bits - 1

    @pytest.mark.parametrize(
        ("kernel", "bits"),
        [(2, 2), (2, 4), (2, 8), (3, 2), (3, 3), (3, 4), (3, 8), ((8, 8), 4), ((5, 3), 3)],
    )
    def test_trunc_avg_pool_matches_brevitas(
        self, kernel: int | tuple[int, int], bits: int
    ) -> None:
        """Output values and the statically derived output format agree with what Brevitas
        itself reports for the pooled quant tensor."""
        act, fmt = build.quant_activation(bits)
        warm_up(act, torch.rand(64) * 7)
        scale = float(fmt.scale())
        pool = build.TruncAvgPool(kernel, fmt).eval()
        x = torch.randint(0, 2**bits, (2, 3, 24, 24)).float() * scale
        y = pool(x)
        pool.pool.return_quant_tensor = True
        ref = pool.pool(build.IntQuantTensor(x, scale, 0.0, float(bits), False, False))
        assert torch.equal(y, ref.value)
        assert float(ref.bit_width) == bits
        out_scale = float(pool.out_format.scale())
        assert out_scale == pytest.approx(float(ref.scale))
        assert pool.out_format.source is act
        # Integers of the output format, and the (truncated) average within one step.
        assert torch.allclose(y / out_scale, (y / out_scale).round(), atol=1e-4)
        avg = nn.functional.avg_pool2d(x, kernel)
        assert ((avg - y) >= -1e-3).all() and ((avg - y) <= out_scale + 1e-3).all()

    def test_format_tracking(self) -> None:
        """Average pooling is truncating exactly where its input is on a known integer grid."""
        quant_first = G.make_genotype(
            G.sequential(
                G.avgpool(2),
                G.quant_act(4),
                G.maxpool(2),
                G.avgpool(2),
                G.batchnorm(),
                G.avgpool(2),
            ),
            head_kind="gap",
        )
        net = to_module(quant_first)
        body = net.body
        assert isinstance(body[0], build.TruncAvgPool)  # 8-bit input quantizer
        assert body[0].fmt.source is net.input_quant
        assert isinstance(body[3], build.TruncAvgPool)  # quant_act, through max pooling
        assert body[3].fmt.source is body[1]
        assert isinstance(body[5], nn.AvgPool2d)  # batch norm makes it float again
        assert isinstance(net.reduce[0], nn.AdaptiveAvgPool2d)
        gap = G.make_genotype(G.sequential(G.conv(8, 3), G.quant_act(2)), head_kind="gap")
        assert isinstance(to_module(gap).reduce[0], build.TruncAvgPool)
        flat = G.make_genotype(G.sequential(G.conv(8, 3), G.quant_act(2)), head_kind="flatten")
        assert isinstance(to_module(flat).reduce, nn.Flatten)
        # A join puts its operands on the join quantizer's grid: the sum of two operands needs
        # one more bit, a concat keeps the width.
        after_add = G.make_genotype(
            G.sequential(G.quant_act(2), G.branching(G.identity(), G.identity(), join_bits=3)),
            head_kind="gap",
        )
        net = to_module(after_add)
        assert isinstance(net.reduce[0], build.TruncAvgPool)
        fmt = net.reduce[0].fmt
        assert (fmt.bits, fmt.signed, fmt.source) == (4, True, net.body[1].join_quant)
        after_cat = G.make_genotype(
            G.sequential(
                G.quant_act(2), G.branching(G.identity(), G.identity(), agg="concat", join_bits=3)
            ),
            head_kind="gap",
        )
        fmt = to_module(after_cat).reduce[0].fmt
        assert (fmt.bits, fmt.signed) == (3, True)
        bipolar_join = G.make_genotype(
            G.sequential(G.quant_act(2), G.branching(G.identity(), G.identity(), join_bits=1)),
            head_kind="gap",
        )
        assert isinstance(to_module(bipolar_join).reduce[0], nn.AdaptiveAvgPool2d)

    def test_join_operands_share_the_quantizer(self) -> None:
        genotype = G.make_genotype(
            G.sequential(
                G.quant_act(4),
                G.branching(G.identity(), G.sequential(G.batchnorm(), G.quant_act(2)), join_bits=5),
            )
        )
        net = to_module(genotype)
        join = net.body[1]
        assert isinstance(join, Branching)
        x = torch.randn(4, 3, 32, 32) * 3
        warm_up(net, x)
        with torch.no_grad():
            h = net.body[0](net.input_quant(x))
            operands = [join.join_quant(branch(h)) for branch in join.branches]
            expected = operands[0] + operands[1]
            assert torch.equal(net.body(net.input_quant(x)), expected)
        scale = join.out_format.scale()
        assert torch.allclose(expected / scale, (expected / scale).round(), atol=1e-4)


@pytest.mark.util
class TestSeeds:
    """Every seed validates, builds, runs on a (2, 3, 32, 32) batch and exports."""

    @pytest.mark.parametrize("name", sorted(seed_genotypes()))
    def test_seed(self, name: str, tmp_path: Path) -> None:
        from finn.nas.export import export_candidate

        genotype = seed_genotypes()[name]
        SearchSpace().validate(genotype)
        net = to_module(genotype)
        logits = net(torch.randn(2, 3, 32, 32))
        assert logits.shape == (2, 10)
        assert n_weights(net) == count_params(genotype)
        onnx_path = export_candidate(net, (1, 3, 32, 32), tmp_path / f"{name}.onnx")
        assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    def test_seed_structure(self) -> None:
        seeds = seed_genotypes()
        kinds = {
            name: [node["kind"] for _, node in iter_nodes(g["body"]) if "children" not in node]
            for name, g in seeds.items()
        }
        assert kinds["cnv_like"].count("conv") == 6 and kinds["cnv_like"].count("maxpool") == 2
        assert "flatten" in kinds["cnv_like"] and "linear" in kinds["cnv_like"]
        assert kinds["mlp_like"][0] == "flatten" and "conv" not in kinds["mlp_like"]
        joins = [n for _, n in iter_nodes(seeds["residual"]["body"]) if n.get("agg") == "add"]
        assert len(joins) == 1 and joins[0]["join_bits"] == 5
        assert any(n.get("depthwise") for _, n in iter_nodes(seeds["depthwise_separable"]["body"]))
        assert seeds["depthwise_separable"]["head"]["kind"] == "gap"
