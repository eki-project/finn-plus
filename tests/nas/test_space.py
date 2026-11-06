"""Unit tests of the grammar-based search space that need neither torch nor FINN steps."""

from __future__ import annotations

import pytest

import copy
import json
import random
import subprocess
import sys
from pathlib import Path

from finn.nas.space import (
    InvalidGenotypeError,
    SamplingError,
    SearchSpace,
    SpaceConfig,
    canonical_json,
    count_params,
    describe,
)
from finn.nas.space import genotype as G
from finn.nas.space import genotype_hash, infer_shapes, seed_genotypes, validate
from finn.nas.space.genotype import iter_nodes
from finn.nas.space.sampler import terminal_variants

N_SAMPLES = 200

#: A fixed genotype and its sha256, computed independently of the package.
GOLDEN = {
    "version": 1,
    "input_shape": [3, 8, 8],
    "num_classes": 4,
    "input_bits": 8,
    "body": {
        "kind": "sequential",
        "children": [
            {
                "kind": "conv",
                "kernel": 3,
                "stride": 1,
                "padding": "same",
                "depthwise": False,
                "out_channels": 8,
                "weight_bits": 2,
            },
            {"kind": "quant_act", "bits": 2},
        ],
    },
    "head": {"kind": "gap", "weight_bits": 4},
}
GOLDEN_HASH = "395d2de8232cc39e1dae7d1e7b5f8935b666f435ecb2f857b4318606f9bcddfb"

#: Run in a fresh interpreter. The parent packages are stubbed so that only the space package
#: itself is executed: ``finn.nas`` imports the oracle, which legitimately needs torch.
ISOLATED_IMPORT = """
import json, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
for name, path in (("finn", root), ("finn.nas", root / "nas")):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
import random
import finn.nas.space as space
genotype = space.SearchSpace().sample(random.Random(7))
child = space.SearchSpace().mutate(genotype, random.Random(8))
print(json.dumps({
    "heavy": sorted(m for m in ("torch", "brevitas", "numpy", "qonnx") if m in sys.modules),
    "golden": space.genotype_hash(json.loads(sys.argv[2])),
    "sample": space.genotype_hash(genotype),
    "child": space.genotype_hash(child),
}))
"""


def small_config(**overrides: object) -> SpaceConfig:
    """Return a config with a small parameter budget so that tests stay fast."""
    return SpaceConfig(**{"max_params": 50_000, "max_nodes": 30, **overrides})


def samples(space: SearchSpace, n: int = N_SAMPLES, offset: int = 0) -> list[dict]:
    """Draw ``n`` genotypes with the seeds ``offset .. offset + n - 1``."""
    return [space.sample(random.Random(offset + seed)) for seed in range(n)]


def image(*children: dict, **kwargs: object) -> dict:
    """Wrap ``children`` into a genotype for a 3x8x8 input with 4 classes."""
    return G.make_genotype(G.sequential(*children), input_shape=(3, 8, 8), num_classes=4, **kwargs)


@pytest.mark.util
class TestGenotype:
    """Canonical form, hashing and JSON round trip."""

    def test_golden_hash(self) -> None:
        assert genotype_hash(GOLDEN) == GOLDEN_HASH
        assert image(G.conv(8, 3, weight_bits=2), G.quant_act(2), head_weight_bits=4) == GOLDEN

    def test_canonical_json_ignores_key_order(self) -> None:
        shuffled = {key: GOLDEN[key] for key in reversed(list(GOLDEN))}
        assert list(shuffled) != list(GOLDEN)
        assert canonical_json(shuffled) == canonical_json(GOLDEN)
        assert " " not in canonical_json(GOLDEN)

    def test_json_round_trip(self) -> None:
        space = SearchSpace(small_config())
        for genotype in [*samples(space, 50), *seed_genotypes().values()]:
            restored = json.loads(json.dumps(genotype))
            assert restored == genotype
            assert genotype_hash(restored) == genotype_hash(genotype)
            assert json.loads(canonical_json(genotype)) == genotype

    def test_hash_is_stable_across_processes_and_package_is_torch_free(self) -> None:
        """A fresh interpreter with another hash seed gets the same hashes and samples, and
        importing/sampling/mutating pulls in neither torch nor any other heavy dependency."""
        src = Path(sys.modules["finn.nas.space"].__file__).parents[2]
        proc = subprocess.run(
            [sys.executable, "-c", ISOLATED_IMPORT, str(src), json.dumps(GOLDEN)],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": "12345"},
        )
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        assert report["heavy"] == []
        assert report["golden"] == GOLDEN_HASH
        space = SearchSpace()
        genotype = space.sample(random.Random(7))
        assert report["sample"] == genotype_hash(genotype)
        assert report["child"] == genotype_hash(space.mutate(genotype, random.Random(8)))

    def test_config_round_trip(self) -> None:
        config = small_config(input_shape=(1, 28, 28), terminal_weights={"conv": 1.0})
        restored = SpaceConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert restored == config
        assert restored.input_shape == (1, 28, 28)
        assert SpaceConfig.from_dict({**config.to_dict(), "unknown": 1}) == config

    def test_config_rejects_unknown_productions(self) -> None:
        with pytest.raises(ValueError, match="terminal_weights"):
            SpaceConfig(terminal_weights={"conv": 1.0, "attention": 1.0})
        with pytest.raises(ValueError, match="head kinds"):
            SpaceConfig(head_kinds=["gap", "topk"])


@pytest.mark.util
class TestShapes:
    """Analytic shape inference and the well-formedness rules."""

    def test_conv_and_pool_arithmetic(self) -> None:
        genotype = image(
            G.conv(8, 3, stride=2, padding="same"),  # ceil(8 / 2) = 4
            G.conv(16, 3, padding="valid"),  # 4 - 2 = 2
            G.depthwise_conv(1),
            G.maxpool(2, 1),  # 2 - 1 = 1
        )
        info = infer_shapes(genotype)
        assert [info.nodes[(i,)].out_shape for i in range(4)] == [
            (8, 4, 4),
            (16, 2, 2),
            (16, 2, 2),
            (16, 1, 1),
        ]
        assert [info.nodes[(i,)].params for i in range(4)] == [3 * 8 * 9, 8 * 16 * 9, 16, 0]
        assert info.nodes[()].in_shape == (3, 8, 8)
        assert info.body_out == (16, 1, 1)
        assert info.output_shape == (4,)
        assert count_params(genotype) == 216 + 1152 + 16 + 16 * 4

    def test_vector_mode_and_batchnorm(self) -> None:
        genotype = image(G.batchnorm(), G.avgpool(2), G.flatten(), G.linear(16), G.batchnorm())
        info = infer_shapes(genotype)
        assert info.nodes[(1,)].out_shape == (3, 4, 4)
        assert info.nodes[(2,)].out_shape == (48,)
        assert info.body_out == (16,)
        assert count_params(genotype) == 2 * 3 + 48 * 16 + 2 * 16 + 16 * 4

    def test_aggregations(self) -> None:
        skip = image(G.branching(G.identity(), G.conv(3, 3), agg="add"))
        assert infer_shapes(skip).body_out == (3, 8, 8)
        concat = image(G.branching(G.conv(8, 3), G.conv(5, 1), G.identity(), agg="concat"))
        assert infer_shapes(concat).body_out == (16, 8, 8)
        vectors = image(
            G.flatten(), G.branching(G.linear(16), G.identity(), agg="concat"), G.linear(32)
        )
        info = infer_shapes(vectors)
        assert info.nodes[(1,)].out_shape == (16 + 192,)
        assert info.nodes[(1,)].size == 3 and info.num_nodes == 6 and info.depth == 2

    def test_head_features(self) -> None:
        body = [G.conv(8, 3)]
        assert infer_shapes(image(*body, head_kind="gap")).head_features == 8
        assert infer_shapes(image(*body, head_kind="flatten")).head_features == 8 * 8 * 8
        # After a flatten there is nothing left to pool: the head kind is a neutral gene.
        assert infer_shapes(image(G.flatten(), head_kind="gap")).head_features == 192

    @pytest.mark.parametrize(
        ("genotype", "reason"),
        [
            (image(G.conv(8, 5, padding="valid"), G.conv(8, 5, padding="valid")), "does not fit"),
            (image(G.maxpool(3, 1), G.maxpool(3, 1), G.maxpool(3, 1), G.avgpool(3)), "not fit"),
            (image(G.linear(16)), "linear needs a vector"),
            (image(G.flatten(), G.flatten()), "flatten needs a feature map"),
            (image(G.flatten(), G.conv(8, 1)), "conv needs a feature map"),
            (image(G.branching(G.identity(), G.conv(8, 3), agg="add")), "equal branch shapes"),
            (image(G.branching(G.identity(), G.maxpool(2), agg="concat")), "equal mode and H, W"),
            (image(G.branching(G.identity(), G.flatten(), agg="concat")), "equal mode and H, W"),
            (image(G.branching(G.identity(), G.identity(), agg="mul")), "unknown aggregation"),
            (image(G.branching(G.identity())), "at least 2 children"),
            (image({"kind": "attention"}), "unknown node kind"),
            (image({"kind": "quant_act"}), "must have fields"),
            (image({**G.depthwise_conv(3), "out_channels": 8}), "must have fields"),
            (image({**G.conv(8, 3), "in_channels": 3}), "must have fields"),
            (image({**G.conv(8, 3), "stride": True}), "stride must be int"),
            (image({**G.conv(8, 3), "padding": "full"}), "padding must be one of"),
            (image(G.quant_act(0)), "bits must be >= 1"),
            (image(G.conv(8, 3), head_kind="topk"), "head must be"),
            ({**image(G.identity()), "version": 2}, "unsupported genotype version"),
            ({**image(G.identity()), "shapes": []}, "genotype must have keys"),
            ({**image(G.identity()), "input_shape": [3, 8]}, "input_shape must be"),
            ({**image(G.identity()), "body": {"kind": "sequential", "children": []}}, "at least"),
        ],
    )
    def test_ill_formed_genotypes_are_rejected(self, genotype: dict, reason: str) -> None:
        with pytest.raises(InvalidGenotypeError, match=reason):
            validate(genotype)

    def test_error_names_the_node(self) -> None:
        genotype = image(G.identity(), G.sequential(G.identity(), G.linear(16)))
        with pytest.raises(InvalidGenotypeError, match=r"node \[1, 1\]: linear needs a vector"):
            infer_shapes(genotype)

    def test_membership_in_a_config(self) -> None:
        cifar = G.make_genotype(G.sequential(G.conv(16, 3), G.quant_act(2)))
        validate(cifar, SpaceConfig())
        for overrides, reason in (
            ({"max_params": 100}, "params"),
            ({"max_nodes": 2}, "nodes"),
            ({"max_depth": 0}, "depth"),
            ({"input_shape": (1, 28, 28)}, "task mismatch"),
            ({"num_classes": 100}, "task mismatch"),
            ({"conv_out_channels": [8, 32]}, "out_channels=16 is not in the space"),
            ({"act_bits": [1]}, "bits=2 is not in the space"),
            ({"terminal_weights": {"conv": 1.0}}, "quant_act is not in the space"),
            ({"head_kinds": ["flatten"]}, "head.kind='gap' is not in the space"),
            ({"conv_depthwise_prob": 1.0}, "depthwise=False is not in the space"),
        ):
            with pytest.raises(InvalidGenotypeError, match=reason):
                validate(cifar, SpaceConfig(**overrides))

    def test_describe(self) -> None:
        genotype = seed_genotypes()["residual"]
        text = describe(genotype)
        lines = text.splitlines()
        info = infer_shapes(genotype)
        # Header, input quantizer, one line per node, head.
        assert len(lines) == info.num_nodes + 3
        assert genotype_hash(genotype)[:12] in lines[0] and str(info.params) in lines[0]
        assert "  branching add => 16x32x32" in lines
        assert "        conv k3 s1 same c16 w4 -> 16x32x32" in lines
        assert lines[-1] == "head flatten linear f10 w4 -> 10"
        assert "depthwise" in describe(seed_genotypes()["depthwise_separable"])


@pytest.mark.util
class TestSampling:
    """The probabilistic grammar."""

    def test_deterministic_for_a_seed(self) -> None:
        space = SearchSpace(small_config())
        first = [genotype_hash(g) for g in samples(space, 30)]
        # Neither the global random state nor earlier calls may matter.
        random.seed(999)
        random.random()
        again = [genotype_hash(g) for g in samples(SearchSpace(small_config()), 30)]
        assert first == again
        assert len(set(first)) > 25
        rng = random.Random(3)
        sequence = [genotype_hash(space.sample(rng)) for _ in range(5)]
        rng = random.Random(3)
        assert sequence == [genotype_hash(space.sample(rng)) for _ in range(5)]

    def test_samples_are_members_of_the_space(self) -> None:
        config = small_config()
        space = SearchSpace(config)
        for genotype in samples(space):
            info = space.validate(genotype)
            assert info.params <= config.max_params
            assert info.num_nodes <= config.max_nodes
            assert info.depth <= config.max_depth
            assert genotype["body"]["kind"] == "sequential"
        default = SearchSpace()
        for genotype in samples(default, 50):
            default.validate(genotype)

    def test_all_productions_occur(self) -> None:
        """Every terminal, both aggregations, skip connections and both modes are reachable."""
        kinds: set[str] = set()
        aggs: set[str] = set()
        skips = vector_outputs = 0
        space = SearchSpace(small_config())
        for genotype in samples(space):
            for _, node in iter_nodes(genotype["body"]):
                kinds.add(node["kind"])
                if node["kind"] == "branching":
                    aggs.add(node["agg"])
                    skips += any(child["kind"] == "identity" for child in node["children"])
            vector_outputs += len(infer_shapes(genotype).body_out) == 1
        assert kinds == {"sequential", "branching", *space.config.terminal_weights}
        assert aggs == {"add", "concat"}
        assert skips > 0
        assert 0 < vector_outputs < N_SAMPLES

    def test_budgets_shape_the_samples(self) -> None:
        flat = SearchSpace(small_config(max_depth=1))
        assert all(flat.validate(g).depth == 1 for g in samples(flat, 30))
        tiny = SearchSpace(small_config(max_nodes=3))
        assert all(tiny.validate(g).num_nodes == 3 for g in samples(tiny, 30))
        only_conv = SearchSpace(small_config(terminal_weights={"conv": 1.0, "quant_act": 1.0}))
        for genotype in samples(only_conv, 30):
            kinds = {node["kind"] for _, node in iter_nodes(genotype["body"])}
            assert kinds <= {"sequential", "branching", "conv", "quant_act"}

    def test_sampling_error_when_nothing_fits(self) -> None:
        # Even the smallest head (3 features x 10 classes) exceeds this parameter budget.
        space = SearchSpace(SpaceConfig(max_params=1, max_sample_attempts=5))
        with pytest.raises(SamplingError):
            space.sample(random.Random(0))

    def test_terminal_variants_follow_the_choice_lists(self) -> None:
        config = SpaceConfig()
        variants = terminal_variants(config, "conv")
        # kernel x stride x padding x weight bits, each dense (per out_channels) or depthwise.
        assert len(variants) == 3 * 2 * 2 * 5 * (5 + 1)
        depthwise = sum(weight for node, weight in variants if node["depthwise"])
        total = sum(weight for _, weight in variants)
        assert depthwise / total == pytest.approx(config.conv_depthwise_prob)
        assert all("out_channels" not in node for node, _ in variants if node["depthwise"])
        assert len(terminal_variants(config, "maxpool")) == 4
        assert terminal_variants(config, "identity") == [({"kind": "identity"}, 1.0)]


@pytest.mark.util
class TestVariation:
    """Mutation and crossover."""

    def test_mutations_validate_and_leave_the_parent_alone(self) -> None:
        space = SearchSpace(small_config())
        rng = random.Random(1)
        changed = 0
        for parent in samples(space):
            before = copy.deepcopy(parent)
            child = space.mutate(parent, rng)
            assert parent == before
            space.validate(child)
            assert child is not parent
            changed += genotype_hash(child) != genotype_hash(parent)
        assert changed == N_SAMPLES

    @pytest.mark.parametrize("subtree_prob", [0.0, 1.0])
    def test_both_mutation_kinds(self, subtree_prob: float) -> None:
        space = SearchSpace(SpaceConfig(subtree_mutation_prob=subtree_prob))
        rng = random.Random(2)
        for parent in [*samples(space, 60), *seed_genotypes().values()]:
            child = space.mutate(parent, rng)
            space.validate(child)
            assert genotype_hash(child) != genotype_hash(parent)
            same_structure = [n["kind"] for _, n in iter_nodes(child["body"])] == [
                n["kind"] for _, n in iter_nodes(parent["body"])
            ]
            if subtree_prob == 0.0:
                # A hyperparameter perturbation changes exactly one gene.
                assert same_structure
                assert _differing_genes(parent, child) == 1

    def test_mutation_is_deterministic(self) -> None:
        space = SearchSpace()
        parent = seed_genotypes()["cnv_like"]
        children = [space.mutate(parent, random.Random(5)) for _ in range(2)]
        assert children[0] == children[1]

    def test_mutation_of_an_invalid_parent_is_an_error(self) -> None:
        space = SearchSpace(small_config())
        with pytest.raises(InvalidGenotypeError):
            space.mutate(G.make_genotype(G.sequential(G.linear(16))), random.Random(0))

    def test_crossovers_validate_and_leave_the_parents_alone(self) -> None:
        space = SearchSpace(small_config())
        rng = random.Random(3)
        population = samples(space, N_SAMPLES + 1)
        exchanged = 0
        for a, b in zip(population, population[1:], strict=False):
            before = copy.deepcopy((a, b))
            child_a, child_b = space.crossover(a, b, rng)
            assert (a, b) == before
            space.validate(child_a)
            space.validate(child_b)
            parents = {genotype_hash(a), genotype_hash(b)}
            assert genotype_hash(child_a) not in parents
            assert genotype_hash(child_b) not in parents
            exchanged += _is_exchange(a, b, child_a, child_b)
        # The fallback (two mutations) is the exception, not the rule.
        assert exchanged > 0.9 * N_SAMPLES

    def test_crossover_of_seeds(self) -> None:
        space = SearchSpace()
        seeds = seed_genotypes()
        rng = random.Random(4)
        for a in seeds.values():
            for b in seeds.values():
                for child in space.crossover(a, b, rng):
                    space.validate(child)


def _differing_genes(a: dict, b: dict) -> int:
    """Count the hyperparameters (body and wrapper) in which two same-structure genotypes
    differ; a depthwise flip and the out_channels it adds or removes count as one."""
    count = int(a["input_bits"] != b["input_bits"])
    count += sum(a["head"][key] != b["head"][key] for key in ("kind", "weight_bits"))
    for (_, node_a), (_, node_b) in zip(iter_nodes(a["body"]), iter_nodes(b["body"]), strict=True):
        if "children" in node_a:
            continue
        keys = (set(node_a) | set(node_b)) - {"out_channels"}
        differing = sum(node_a.get(key) != node_b.get(key) for key in keys)
        if not differing:
            differing = node_a.get("out_channels") != node_b.get("out_channels")
        count += differing
    return count


def _is_exchange(a: dict, b: dict, child_a: dict, child_b: dict) -> bool:
    """Return whether the children are the parents with one subtree swapped: together they
    then hold exactly the nodes of both parents."""

    def kinds(*genotypes: dict) -> list[str]:
        return sorted(
            canonical_json(node)
            for genotype in genotypes
            for _, node in iter_nodes(genotype["body"])
            if "children" not in node
        )

    return kinds(a, b) == kinds(child_a, child_b)
