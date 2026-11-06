"""Configuration of the grammar-based search space: task, budgets, production weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

#: The three productions of the ``module`` non-terminal.
MODULE_KINDS = ("sequential", "branching", "terminal")
#: Ways to merge the branches of a ``branching`` node.
AGGREGATIONS = ("add", "concat")
#: Terminal operations of the grammar.
TERMINAL_KINDS = (
    "conv",
    "linear",
    "quant_act",
    "relu_float",
    "batchnorm",
    "maxpool",
    "avgpool",
    "flatten",
    "identity",
)
#: Kinds of the classification head.
HEAD_KINDS = ("gap", "flatten")


def _default_module_weights() -> dict[str, float]:
    """Return the default production weights of ``module``.

    With on average three children per sequential node and two per branching node these
    weights give slightly more than one child per node, i.e. the recursion would not die out
    by itself; the depth and node budgets bound it. That keeps the typical body at 10-25
    nodes instead of being dominated by trivial two-layer networks.
    """
    return {"sequential": 0.3, "branching": 0.15, "terminal": 0.55}


def _default_terminal_weights() -> dict[str, float]:
    """Return the default production weights of ``terminal``.

    They roughly follow how often the operations occur in hand-designed quantized networks
    (computation and activation dominate, shape changes are rarer). They deliberately encode
    no FINN knowledge: ``relu_float`` and ``identity`` are as likely as a pooling layer.
    """
    return {
        "conv": 3.0,
        "linear": 3.0,
        "quant_act": 3.0,
        "batchnorm": 2.0,
        "relu_float": 1.0,
        "maxpool": 1.0,
        "avgpool": 1.0,
        "flatten": 0.5,
        "identity": 1.0,
    }


@dataclass
class SpaceConfig:
    """Everything that defines a search space; JSON-serializable via :meth:`to_dict`.

    The defaults describe small CIFAR-10 classifiers. Weights are relative (they need not sum
    to one); a weight of zero removes a production. Choice lists are sampled uniformly.
    """

    #: Task: ``(C, H, W)`` of the network input and the number of output logits.
    input_shape: tuple[int, int, int] = (3, 32, 32)
    num_classes: int = 10

    #: Maximum nesting depth: a node whose path from the body root has this length must be a
    #: terminal. 6 allows e.g. a residual block inside a stage inside a branch.
    max_depth: int = 6
    #: Maximum number of tree nodes of the body (inner nodes and terminals). CNV has 28
    #: operations, so 40 leaves room for some structure around a network of that size.
    max_nodes: int = 40
    #: Maximum number of trainable parameters (see ``count_params``). CNV has 1.5 M.
    max_params: int = 2_000_000

    module_weights: dict[str, float] = field(default_factory=_default_module_weights)
    #: Number of children of a sampled sequential node.
    sequential_lengths: list[int] = field(default_factory=lambda: [2, 3, 4])
    #: Number of branches of a sampled branching node.
    branch_counts: list[int] = field(default_factory=lambda: [2, 3])
    agg_weights: dict[str, float] = field(default_factory=lambda: {"add": 1.0, "concat": 1.0})
    terminal_weights: dict[str, float] = field(default_factory=_default_terminal_weights)

    conv_kernels: list[int] = field(default_factory=lambda: [1, 3, 5])
    conv_strides: list[int] = field(default_factory=lambda: [1, 2])
    conv_out_channels: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    conv_paddings: list[str] = field(default_factory=lambda: ["same", "valid"])
    #: Probability that a sampled convolution is depthwise.
    conv_depthwise_prob: float = 0.25
    linear_out_features: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    #: Weight bit widths of conv and linear layers (1 = bipolar).
    weight_bits: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])
    #: Activation bit widths (1 = bipolar sign, otherwise unsigned quantized ReLU).
    act_bits: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])
    maxpool_kernels: list[int] = field(default_factory=lambda: [2, 3])
    maxpool_strides: list[int] = field(default_factory=lambda: [1, 2])
    avgpool_kernels: list[int] = field(default_factory=lambda: [2, 3])

    #: Bit widths of the fixed input quantizer.
    input_bits: list[int] = field(default_factory=lambda: [8])
    head_kinds: list[str] = field(default_factory=lambda: ["gap", "flatten"])
    head_weight_bits: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])

    #: Complete restarts of ``sample`` before ``SamplingError`` is raised.
    max_sample_attempts: int = 100
    #: Rejection-sampling attempts for every later branch of a branching node to end in a
    #: shape that can be aggregated with the first branch.
    branch_match_attempts: int = 20
    #: Attempts of ``mutate`` / ``crossover`` to produce a valid, different genotype.
    max_variation_attempts: int = 100
    #: ``mutate`` resamples a subtree with this probability and otherwise perturbs one
    #: hyperparameter.
    subtree_mutation_prob: float = 0.5

    def __post_init__(self) -> None:
        """Normalize the task shape and reject unknown production names early."""
        self.input_shape = tuple(self.input_shape)
        for name, weights, known in (
            ("module_weights", self.module_weights, MODULE_KINDS),
            ("agg_weights", self.agg_weights, AGGREGATIONS),
            ("terminal_weights", self.terminal_weights, TERMINAL_KINDS),
        ):
            unknown = sorted(set(weights) - set(known))
            if unknown:
                raise ValueError(f"Unknown entries {unknown} in {name}; known: {list(known)}.")
        unknown = sorted(set(self.head_kinds) - set(HEAD_KINDS))
        if unknown:
            raise ValueError(f"Unknown head kinds {unknown}; known: {list(HEAD_KINDS)}.")

    def choices(self, kind: str) -> dict[str, list[Any]]:
        """Return the choice list of every hyperparameter of terminal ``kind``.

        This is the single place that ties genotype fields to config fields; sampling,
        hyperparameter mutation and the membership check all go through it.
        """
        if kind == "conv":
            prob = self.conv_depthwise_prob
            depthwise = [flag for flag, p in ((False, 1.0 - prob), (True, prob)) if p > 0]
            return {
                "kernel": self.conv_kernels,
                "stride": self.conv_strides,
                "padding": self.conv_paddings,
                "weight_bits": self.weight_bits,
                "depthwise": depthwise,
                "out_channels": self.conv_out_channels,
            }
        if kind == "linear":
            return {"out_features": self.linear_out_features, "weight_bits": self.weight_bits}
        if kind == "quant_act":
            return {"bits": self.act_bits}
        if kind == "maxpool":
            return {"kernel": self.maxpool_kernels, "stride": self.maxpool_strides}
        if kind == "avgpool":
            return {"kernel": self.avgpool_kernels}
        return {}

    def wrapper_choices(self) -> dict[str, list[Any]]:
        """Return the choice lists of the genes outside the body (input quantizer and head)."""
        return {
            "input_bits": self.input_bits,
            "head.kind": self.head_kinds,
            "head.weight_bits": self.head_weight_bits,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpaceConfig:
        """Reconstruct a config from :meth:`to_dict` output, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
