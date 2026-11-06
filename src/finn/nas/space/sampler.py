"""Recursive sampling from the probabilistic grammar, with shape propagation and backtracking.

Grammar (weights and choice lists come from :class:`SpaceConfig`)::

    body     -> sequential
    module   -> sequential | branching | terminal
    sequential -> module module [module [module]]
    branching  -> (module, module [, module]; add | concat)
    terminal -> conv | linear | quant_act | relu_float | batchnorm | maxpool | avgpool
              | flatten | identity

Like einspace, the sampler carries the current shape down the recursion, only offers the
productions that are applicable to it and to the remaining budget, and backtracks when a
production turns out to be a dead end. Unlike einspace nothing derived is stored in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, TypeVar

from finn.nas.space.genotype import VERSION, Genotype, InvalidGenotypeError, Node
from finn.nas.space.shapes import Shape, aggregate, apply_terminal, validate

if TYPE_CHECKING:
    import random

    from finn.nas.space.config import SpaceConfig

T = TypeVar("T")


class SamplingError(RuntimeError):
    """No valid genotype was found within the configured number of attempts."""


class DeadEndError(Exception):
    """No production is applicable at the current position; the caller must backtrack."""


@dataclass(frozen=True)
class Sampled:
    """A sampled subtree together with what the caller needs to continue."""

    node: Node
    out_shape: Shape
    #: Number of tree nodes of the subtree.
    size: int
    #: Trainable parameters of the subtree.
    params: int


def weighted_choice(rng: random.Random, weights: dict[T, float]) -> T:
    """Draw a key of ``weights`` with probability proportional to its value."""
    keys = list(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


def terminal_variants(config: SpaceConfig, kind: str) -> list[tuple[Node, float]]:
    """Enumerate all hyperparameter settings of terminal ``kind`` with their weights.

    Choice lists are uniform, so every variant weighs 1. The exception is conv: a depthwise
    convolution has no ``out_channels``, so its single variant carries the weight of all the
    dense variants it stands for, scaled by the depthwise probability.
    """
    choices = dict(config.choices(kind))
    if kind != "conv":
        names = list(choices)
        return [
            ({"kind": kind, **dict(zip(names, values, strict=True))}, 1.0)
            for values in product(*choices.values())
        ]
    out_channels = choices.pop("out_channels")
    choices.pop("depthwise")
    names = list(choices)
    prob = config.conv_depthwise_prob
    variants: list[tuple[Node, float]] = []
    for values in product(*choices.values()):
        base = {"kind": "conv", **dict(zip(names, values, strict=True))}
        if prob > 0:
            variants.append(({**base, "depthwise": True}, prob))
        if prob < 1:
            weight = (1.0 - prob) / len(out_channels)
            variants.extend(
                ({**base, "depthwise": False, "out_channels": c}, weight) for c in out_channels
            )
    return variants


class Sampler:
    """Draws genotypes and subtrees from the grammar defined by a :class:`SpaceConfig`."""

    def __init__(self, config: SpaceConfig) -> None:
        """Precompute the hyperparameter variants of all enabled terminals."""
        self.config = config
        self._terminal_weights = {k: w for k, w in config.terminal_weights.items() if w > 0}
        self._variants = {kind: terminal_variants(config, kind) for kind in self._terminal_weights}
        self._applicable_cache: dict[tuple[str, Shape], dict[int, tuple[float, Shape, int]]] = {}

    def sample(self, rng: random.Random) -> Genotype:
        """Draw a genotype that validates against the config.

        The body is restarted from scratch when it is a dead end or when the complete network
        (body plus head) exceeds a budget; :class:`SamplingError` is raised only after
        ``max_sample_attempts`` such restarts.
        """
        cfg = self.config
        for _ in range(cfg.max_sample_attempts):
            try:
                body = self.body(rng)
            except DeadEndError:
                continue
            genotype = {
                "version": VERSION,
                "input_shape": list(cfg.input_shape),
                "num_classes": cfg.num_classes,
                "input_bits": rng.choice(cfg.input_bits),
                "body": body.node,
                "head": {
                    "kind": rng.choice(cfg.head_kinds),
                    "weight_bits": rng.choice(cfg.head_weight_bits),
                },
            }
            try:
                validate(genotype, cfg)
            except InvalidGenotypeError:
                continue
            return genotype
        raise SamplingError(f"No valid genotype in {cfg.max_sample_attempts} attempts.")

    def body(self, rng: random.Random) -> Sampled:
        """Sample a body root. It is always a sequential node: a body consisting of a single
        operation is not an interesting start for a search."""
        cfg = self.config
        return self._sequential(rng, tuple(cfg.input_shape), 0, cfg.max_nodes, cfg.max_params)

    def module(
        self, rng: random.Random, shape: Shape, depth: int, nodes_left: int, params_left: int
    ) -> Sampled:
        """Sample a ``module`` for input ``shape`` at nesting ``depth`` within the budgets.

        Productions that cannot fit the budget are not offered; a production that fails is
        removed and another one is drawn (backtracking). Raises :class:`DeadEndError` when
        none is left.
        """
        samplers = {
            "sequential": self._sequential,
            "branching": self._branching,
            "terminal": self._terminal,
        }
        options = {
            kind: weight
            for kind, weight in self.config.module_weights.items()
            if weight > 0 and self._min_size(kind, depth) <= nodes_left
        }
        while options:
            kind = weighted_choice(rng, options)
            try:
                return samplers[kind](rng, shape, depth, nodes_left, params_left)
            except DeadEndError:
                del options[kind]
        raise DeadEndError

    def _min_size(self, kind: str, depth: int) -> float:
        """Return the smallest number of tree nodes a ``kind`` production needs at ``depth``."""
        if kind == "terminal":
            return 1
        if depth >= self.config.max_depth:
            return float("inf")
        counts = (
            self.config.sequential_lengths if kind == "sequential" else self.config.branch_counts
        )
        return 1 + min(counts)

    def _sequential(
        self, rng: random.Random, shape: Shape, depth: int, nodes_left: int, params_left: int
    ) -> Sampled:
        """Sample a chain; every child sees the output shape of its predecessor."""
        lengths = [n for n in self.config.sequential_lengths if n + 1 <= nodes_left]
        if not lengths:
            raise DeadEndError
        length = rng.choice(lengths)
        children: list[Node] = []
        size, params = 1, 0
        for index in range(length):
            # Keep one node in reserve for each sibling that is still to come.
            reserve = length - index - 1
            child = self.module(
                rng, shape, depth + 1, nodes_left - size - reserve, params_left - params
            )
            children.append(child.node)
            shape, size, params = child.out_shape, size + child.size, params + child.params
        return Sampled({"kind": "sequential", "children": children}, shape, size, params)

    def _branching(
        self, rng: random.Random, shape: Shape, depth: int, nodes_left: int, params_left: int
    ) -> Sampled:
        """Sample parallel branches of the same input and an aggregation.

        The first branch is free; each later branch is rejection-sampled until its output can
        be aggregated with the earlier ones (equal shapes for add, equal H and W for concat).
        """
        counts = [n for n in self.config.branch_counts if n + 1 <= nodes_left]
        if not counts:
            raise DeadEndError
        count = rng.choice(counts)
        agg = weighted_choice(rng, {a: w for a, w in self.config.agg_weights.items() if w > 0})
        branches: list[Sampled] = []
        size, params = 1, 0
        for index in range(count):
            reserve = count - index - 1
            for _ in range(self.config.branch_match_attempts):
                branch = self.module(
                    rng, shape, depth + 1, nodes_left - size - reserve, params_left - params
                )
                try:
                    aggregate(agg, [b.out_shape for b in branches] + [branch.out_shape])
                except InvalidGenotypeError:
                    continue
                break
            else:
                raise DeadEndError
            branches.append(branch)
            size, params = size + branch.size, params + branch.params
        out_shape = aggregate(agg, [b.out_shape for b in branches])
        node = {"kind": "branching", "agg": agg, "children": [b.node for b in branches]}
        return Sampled(node, out_shape, size, params)

    def _terminal(
        self, rng: random.Random, shape: Shape, _depth: int, _nodes_left: int, params_left: int
    ) -> Sampled:
        """Sample an operation and its hyperparameters among those applicable to ``shape``."""
        kinds = dict(self._terminal_weights)
        while kinds:
            kind = weighted_choice(rng, kinds)
            fitting = {
                index: weight
                for index, (weight, _, params) in self._applicable(kind, shape).items()
                if params <= params_left
            }
            if fitting:
                index = weighted_choice(rng, fitting)
                _, out_shape, params = self._applicable(kind, shape)[index]
                return Sampled(dict(self._variants[kind][index][0]), out_shape, 1, params)
            del kinds[kind]
        raise DeadEndError

    def _applicable(self, kind: str, shape: Shape) -> dict[int, tuple[float, Shape, int]]:
        """Return ``{variant index: (weight, output shape, params)}`` of the variants of
        ``kind`` that are well-formed on ``shape``. Cached: few distinct shapes occur."""
        key = (kind, shape)
        if key not in self._applicable_cache:
            entry: dict[int, tuple[float, Shape, int]] = {}
            for index, (node, weight) in enumerate(self._variants[kind]):
                try:
                    out_shape, params = apply_terminal(node, shape)
                except InvalidGenotypeError:
                    continue
                entry[index] = (weight, out_shape, params)
            self._applicable_cache[key] = entry
        return self._applicable_cache[key]
