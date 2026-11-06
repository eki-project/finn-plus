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
    """Build the network of ``genotype`` and evaluate it."""
    torch.manual_seed(0)
    return oracle.evaluate(to_module(genotype).eval(), (1, *genotype["input_shape"]))


def _block(layer: dict, act_bits: int = 4) -> dict:
    """Return ``layer`` followed by batch normalization and a quantized activation."""
    return G.sequential(layer, G.batchnorm(), G.quant_act(act_bits))


@pytest.mark.fpgadataflow
class TestSeeds:
    """All four seeds are FINN-compatible as written.

    cnv_like and mlp_like map like their BNN-PYNQ originals. residual and depthwise_separable
    needed no sampler rules, but two construction choices in ``build.py`` that were found by
    varying the genotypes: activations with a constant scale of exactly 1 (so that the fork
    and the add of the residual block see plain integers; the default streamlining cannot move
    a scale factor across a fork or a join) and average pooling as Brevitas' truncating
    ``TruncAvgPool2d`` whenever the input is known to be integer (the global average pooling
    of the depthwise seed; ``AdaptiveAvgPool2d`` exports as ``ReduceMean``, for which FINN
    has no conversion).
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
            # The identity skip forks the input quantizer's output, which carries a scale of
            # 2**-7: the scale factor cannot be streamlined past the fork.
            (
                "skip_from_input",
                G.make_genotype(
                    G.sequential(
                        G.branching(G.identity(), _block(G.conv(3, 3)), agg="add"), G.maxpool()
                    ),
                    head_kind="flatten",
                ),
            ),
            # Global average pooling of a float tensor (batch norm output) is a ReduceMean.
            (
                "gap_after_batchnorm",
                G.make_genotype(G.sequential(_block(G.conv(8, 3)), G.conv(8, 3), G.batchnorm())),
            ),
        ],
    )
    def test_known_infeasible_variants(self, name: str, genotype: dict, tmp_path: Path) -> None:
        """Genotype-level neighbours of the seeds that FINN rejects; this is the kind of rule
        the search has to learn, and it documents why the seeds are written as they are."""
        result = evaluate(make_oracle(tmp_path), genotype)
        assert not result.feasible, name
        assert result.failure_category in {
            FailureCategory.STREAMLINE_INCOMPLETE,
            FailureCategory.UNSUPPORTED_OP,
            FailureCategory.UNMAPPED_LAYER,
        }


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
