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
            assert sum(p.numel() for p in net.parameters()) == count_params(genotype)
            assert real_shapes(net, list(info.nodes), x) == {
                path: node.out_shape for path, node in info.nodes.items()
            }

    def test_eval_mode_and_single_image(self) -> None:
        space = SearchSpace(SpaceConfig(max_params=50_000))
        for seed in range(20):
            genotype = space.sample(random.Random(seed))
            net = to_module(genotype).eval()
            with torch.no_grad():
                assert net(torch.randn(1, 3, 32, 32)).shape == (1, 10)

    def test_parameter_definition(self) -> None:
        """Weights of conv/linear and the affine parameters of batch norm, nothing else."""
        genotype = seed_genotypes()["cnv_like"]
        net = to_module(genotype)
        names = {name.rsplit(".", 1)[-1] for name, _ in net.named_parameters()}
        assert names == {"weight", "bias"}
        assert all(
            isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
            for module in net.modules()
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter)
        )
        assert sum(p.numel() for p in net.parameters()) == count_params(genotype)

    def test_invalid_genotype_is_rejected(self) -> None:
        with pytest.raises(InvalidGenotypeError):
            to_module(G.make_genotype(G.sequential(G.linear(16))))


@pytest.mark.util
class TestQuantizers:
    """The fixed-point conventions the FINN mapping relies on."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_activations_are_integers(self, bits: int) -> None:
        act, fmt = build.quant_activation(bits)
        y = act(torch.linspace(-3, 300, 1001))
        assert fmt == build.IntFormat(bits, False, 1.0)
        assert torch.equal(y, y.round())
        assert y.min() == 0 and y.max() == 2**bits - 1
        assert len(torch.unique(y)) == 2**bits

    def test_bipolar_activation(self) -> None:
        act, fmt = build.quant_activation(1)
        assert fmt is None
        y = act(torch.tensor([-2.0, -0.1, 0.1, 2.0]))
        assert set(y.tolist()) == {-1.0, 1.0}

    @pytest.mark.parametrize("bits", [2, 8])
    def test_input_quantizer(self, bits: int) -> None:
        quantizer, fmt = build.input_quantizer(bits)
        step = 2.0 ** (1 - bits)
        assert fmt == build.IntFormat(bits, True, step)
        y = quantizer(torch.linspace(-2, 2, 4001))
        assert y.min() == -1.0 and y.max() == 1.0 - step
        assert torch.allclose(y / step, (y / step).round())
        assert len(torch.unique(y)) == 2**bits

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_weights(self, bits: int) -> None:
        conv = build._conv(G.conv(8, 3, weight_bits=bits), (3, 8, 8))
        nn.init.uniform_(conv.weight, -1.0, 1.0)
        q = conv.quant_weight().value.detach()
        levels = 2.0 if bits == 1 else 2 ** (bits - 1) - 1
        assert torch.allclose(q * levels, (q * levels).round(), atol=1e-6)
        assert q.abs().max() == pytest.approx(1.0)
        levels_used = len(torch.unique(q))
        if bits <= 4:
            assert levels_used == (2 if bits == 1 else 2**bits - 1)
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
        fmt = build.IntFormat(bits, False, 1.0)
        pool = build.TruncAvgPool(kernel, fmt).eval()
        x = torch.randint(0, 2**bits, (2, 3, 24, 24)).float()
        y = pool(x)
        pool.pool.return_quant_tensor = True
        ref = pool.pool(build.IntQuantTensor(x, 1.0, 0.0, float(bits), False, False))
        assert torch.equal(y, ref.value)
        assert float(ref.bit_width) == bits
        assert pool.out_format.scale == pytest.approx(float(ref.scale))
        # Integers of the output format, and the (truncated) average within one step.
        assert torch.allclose(y / pool.out_format.scale, (y / pool.out_format.scale).round())
        avg = nn.functional.avg_pool2d(x, kernel)
        assert ((avg - y) >= -1e-3).all() and ((avg - y) <= pool.out_format.scale + 1e-3).all()

    def test_format_tracking(self) -> None:
        """Average pooling is truncating exactly where its input is known to be integer."""
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
        assert isinstance(body[3], build.TruncAvgPool)  # quant_act, through max pooling
        assert isinstance(body[5], nn.AvgPool2d)  # batch norm makes it float again
        assert isinstance(net.reduce[0], nn.AdaptiveAvgPool2d)
        gap = G.make_genotype(G.sequential(G.conv(8, 3), G.quant_act(2)), head_kind="gap")
        assert isinstance(to_module(gap).reduce[0], build.TruncAvgPool)
        flat = G.make_genotype(G.sequential(G.conv(8, 3), G.quant_act(2)), head_kind="flatten")
        assert isinstance(to_module(flat).reduce, nn.Flatten)
        after_add = G.make_genotype(
            G.sequential(G.quant_act(2), G.branching(G.identity(), G.identity())), head_kind="gap"
        )
        assert isinstance(to_module(after_add).reduce[0], nn.AdaptiveAvgPool2d)


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
        assert sum(p.numel() for p in net.parameters()) == count_params(genotype)
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
        assert any(n.get("agg") == "add" for _, n in iter_nodes(seeds["residual"]["body"]))
        assert any(n.get("depthwise") for _, n in iter_nodes(seeds["depthwise_separable"]["body"]))
        assert seeds["depthwise_separable"]["head"]["kind"] == "gap"
