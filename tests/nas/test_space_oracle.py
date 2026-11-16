"""Seed architectures and random samples of the search space through the FINN+ oracle."""

from __future__ import annotations

import pytest

import random
from pathlib import Path

from finn.nas import EvalResult, FailureCategory, FinnOracle, FoldingSpec, OracleConfig
from finn.nas.space import SearchSpace, SpaceConfig, describe
from finn.nas.space import genotype as G
from finn.nas.space import seed_genotypes

torch = pytest.importorskip("torch")
pytest.importorskip("brevitas")
to_module = pytest.importorskip("finn.nas.space.build").to_module

FPGA_PART = "xc7z020clg400-1"


def make_oracle(tmp_path: Path) -> FinnOracle:
    """The oracle configuration used for all search-space evaluations."""
    return FinnOracle(
        OracleConfig(
            fpga_part=FPGA_PART,
            synth_clk_period_ns=10.0,
            folding=FoldingSpec(target_fps=100),
            isolation="inprocess",
            work_dir=str(tmp_path / "oracle"),
        )
    )


def evaluate(oracle: FinnOracle, genotype: dict) -> EvalResult:
    """Build the network of ``genotype``, calibrate its activation scales and evaluate it."""
    from finn.nas.calibrate import calibrate, synthetic_inputs

    torch.manual_seed(0)
    net = to_module(genotype)
    calibrate(net, synthetic_inputs(tuple(genotype["input_shape"]), 64, seed=0), steps=5)
    return oracle.evaluate(net, (1, *genotype["input_shape"]))


def _block(layer: dict, act_bits: int = 4) -> dict:
    """Return ``layer`` followed by batch normalization and a quantized activation."""
    return G.sequential(layer, G.batchnorm(), G.quant_act(act_bits))


@pytest.mark.fpgadataflow
class TestSeeds:
    """All four seeds are FINN-compatible as written.

    cnv_like and mlp_like map like their BNN-PYNQ originals. residual and depthwise_separable
    needed no sampler rules, but two construction choices in ``build.py`` that were found by
    varying the genotypes: a shared join quantizer in every branching node (so that the add
    of the residual block sees two operands on one integer grid; with Brevitas' learned
    per-tensor scales the operands would otherwise differ and FINN cannot reconcile them in
    hardware) and average pooling as Brevitas' truncating ``TruncAvgPool2d`` whenever the
    input is on a known integer grid (the global average pooling of the depthwise seed;
    ``AdaptiveAvgPool2d`` exports as ``ReduceMean``, for which FINN has no conversion). The
    learned scales at forks and joins also need the oracle's default ``StreamlinePlus``
    streamlining; the builder's default streamlining rejects the residual seed.
    """

    @pytest.mark.parametrize("name", sorted(seed_genotypes()))
    def test_seed_is_feasible(self, name: str, tmp_path: Path) -> None:
        genotype = seed_genotypes()[name]
        result = evaluate(make_oracle(tmp_path), genotype)
        assert result.feasible, (
            result.failure_category,
            result.failure_stage,
            result.failure_message,
            describe(genotype),
        )
        assert result.failure_category == FailureCategory.NONE
        assert result.est_fps is not None and result.est_fps > 0
        assert result.resources_total["LUT"] > 0

    @pytest.mark.parametrize(
        ("name", "genotype"),
        [
            # The identity skip forks the input quantizer's output; the join re-quantizes it.
            (
                "skip_from_input",
                G.make_genotype(
                    G.sequential(
                        G.branching(G.identity(), _block(G.conv(3, 3)), agg="add", join_bits=5),
                        G.maxpool(),
                    ),
                    head_kind="flatten",
                ),
            ),
            # Two conv branches merged by concat (StreamingConcat).
            (
                "concat",
                G.make_genotype(
                    G.sequential(
                        _block(G.conv(16, 3)),
                        G.branching(
                            _block(G.conv(8, 3)), _block(G.conv(8, 1)), agg="concat", join_bits=5
                        ),
                        _block(G.conv(16, 3)),
                    ),
                    head_kind="flatten",
                ),
            ),
        ],
    )
    def test_feasible_join_variants(self, name: str, genotype: dict, tmp_path: Path) -> None:
        """Joins the shared join quantizer makes mappable (both failed with constant scales
        and the default streamlining)."""
        result = evaluate(make_oracle(tmp_path), genotype)
        assert result.feasible, (name, result.failure_category, result.failure_message)

    def test_known_infeasible_variant(self, tmp_path: Path) -> None:
        """Global average pooling of a float tensor (batch norm output) is a ReduceMean; this
        is the kind of rule the search has to learn."""
        genotype = G.make_genotype(G.sequential(_block(G.conv(8, 3)), G.conv(8, 3), G.batchnorm()))
        result = evaluate(make_oracle(tmp_path), genotype)
        assert not result.feasible
        assert result.failure_category in {
            FailureCategory.STREAMLINE_INCOMPLETE,
            FailureCategory.UNSUPPORTED_OP,
            FailureCategory.UNMAPPED_LAYER,
        }

    def test_default_streamlining_rejects_learned_scales_at_joins(self, tmp_path: Path) -> None:
        """Documents why the oracle streamlines with StreamlinePlus by default."""
        oracle = make_oracle(tmp_path)
        oracle.config.streamline = "default"
        result = evaluate(oracle, seed_genotypes()["residual"])
        assert not result.feasible

    def test_verdict_is_independent_of_the_weights(self, tmp_path: Path) -> None:
        """Different weight seeds give the same verdict, hash and LUT estimate."""
        from finn.nas.calibrate import calibrate, synthetic_inputs
        from finn.nas.evaluate import randomize_batchnorm

        oracle = make_oracle(tmp_path)
        genotype = seed_genotypes()["residual"]
        results = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            net = randomize_batchnorm(to_module(genotype), seed)
            calibrate(net, synthetic_inputs((3, 32, 32), 64, seed), steps=5)
            results.append(oracle.evaluate(net, (1, 3, 32, 32)))
        a, b = results
        assert a.feasible and b.feasible
        assert a.candidate_hash == b.candidate_hash
        assert abs(a.resources_total["LUT"] - b.resources_total["LUT"]) <= 8


@pytest.mark.fpgadataflow
@pytest.mark.slow
class TestRandomSamples:
    """The oracle answers every random sample with a consistent result."""

    def test_smoke(self, tmp_path: Path) -> None:
        space = SearchSpace(SpaceConfig(max_params=200_000, max_nodes=20))
        oracle = make_oracle(tmp_path)
        outcomes = []
        for seed in range(20):
            genotype = space.sample(random.Random(seed))
            result = evaluate(oracle, genotype)
            assert isinstance(result, EvalResult)
            assert result.feasible == (result.failure_category == FailureCategory.NONE)
            if result.feasible:
                assert result.failure_stage is None and result.est_fps > 0
            else:
                assert result.failure_stage is not None and result.failure_message
            assert result.timings_s["total"] > 0
            outcomes.append(result.failure_category)
        # No assertion on the feasibility rate: that is what the search is supposed to learn.
        assert len(outcomes) == 20
