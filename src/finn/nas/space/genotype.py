"""Genotype representation: plain nested dicts/lists, their constructors, hashing and editing.

A genotype stores only decisions (structure and hyperparameters). Everything that follows from
them (shapes, input channels, parameter counts) is recomputed by :mod:`finn.nas.space.shapes`,
so a genotype cannot become inconsistent with itself when it is mutated or recombined.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Version of the genotype format, stored in every genotype.
VERSION = 1

Genotype = dict[str, Any]
Node = dict[str, Any]
#: Address of a body node: the child indices on the way down from the body root.
Path = tuple[int, ...]

INNER_KINDS = ("sequential", "branching")


class InvalidGenotypeError(ValueError):
    """The genotype is ill-formed or does not describe a network PyTorch could run."""


def sequential(*children: Node) -> Node:
    """Return a node that applies ``children`` one after another."""
    return {"kind": "sequential", "children": list(children)}


def branching(*children: Node, agg: str = "add") -> Node:
    """Return a node that feeds its input to all ``children`` and merges them with ``agg``."""
    return {"kind": "branching", "agg": agg, "children": list(children)}


def conv(
    out_channels: int, kernel: int = 3, stride: int = 1, padding: str = "same", weight_bits: int = 8
) -> Node:
    """Return a regular (dense) convolution terminal."""
    return {
        "kind": "conv",
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "depthwise": False,
        "out_channels": out_channels,
        "weight_bits": weight_bits,
    }


def depthwise_conv(
    kernel: int = 3, stride: int = 1, padding: str = "same", weight_bits: int = 8
) -> Node:
    """Return a depthwise convolution terminal; it has no ``out_channels`` (= input channels)."""
    return {
        "kind": "conv",
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "depthwise": True,
        "weight_bits": weight_bits,
    }


def linear(out_features: int, weight_bits: int = 8) -> Node:
    """Return a fully connected terminal (vector mode only)."""
    return {"kind": "linear", "out_features": out_features, "weight_bits": weight_bits}


def quant_act(bits: int) -> Node:
    """Return a quantized activation terminal (1 bit: bipolar sign, else unsigned ReLU)."""
    return {"kind": "quant_act", "bits": bits}


def relu_float() -> Node:
    """Return a floating-point ReLU terminal."""
    return {"kind": "relu_float"}


def batchnorm() -> Node:
    """Return a batch normalization terminal."""
    return {"kind": "batchnorm"}


def maxpool(kernel: int = 2, stride: int = 2) -> Node:
    """Return a max pooling terminal."""
    return {"kind": "maxpool", "kernel": kernel, "stride": stride}


def avgpool(kernel: int = 2) -> Node:
    """Return an average pooling terminal (stride = kernel)."""
    return {"kind": "avgpool", "kernel": kernel}


def flatten() -> Node:
    """Return the terminal that switches from image mode to vector mode."""
    return {"kind": "flatten"}


def identity() -> Node:
    """Return the identity terminal (as a branch: a skip connection)."""
    return {"kind": "identity"}


def make_genotype(
    body: Node,
    head_kind: str = "gap",
    head_weight_bits: int = 8,
    input_bits: int = 8,
    input_shape: Sequence[int] = (3, 32, 32),
    num_classes: int = 10,
) -> Genotype:
    """Wrap ``body`` with the task description, the input quantizer and the head genes."""
    return {
        "version": VERSION,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "input_bits": input_bits,
        "body": body,
        "head": {"kind": head_kind, "weight_bits": head_weight_bits},
    }


def canonical_json(genotype: Genotype) -> str:
    """Serialize to the canonical form (sorted keys, no whitespace) used for hashing."""
    return json.dumps(genotype, sort_keys=True, separators=(",", ":"), allow_nan=False)


def genotype_hash(genotype: Genotype) -> str:
    """Return the sha256 hex digest of the canonical JSON; stable across processes."""
    return hashlib.sha256(canonical_json(genotype).encode("utf-8")).hexdigest()


def is_inner(node: Node) -> bool:
    """Return whether ``node`` is a sequential or branching node."""
    return node.get("kind") in INNER_KINDS


def iter_nodes(node: Node, path: Path = ()) -> Iterator[tuple[Path, Node]]:
    """Yield ``(path, node)`` for ``node`` and all its descendants in pre-order."""
    yield path, node
    if is_inner(node):
        for index, child in enumerate(node["children"]):
            yield from iter_nodes(child, (*path, index))


def get_node(genotype: Genotype, path: Path) -> Node:
    """Return the body node at ``path`` (not a copy)."""
    node = genotype["body"]
    for index in path:
        node = node["children"][index]
    return node


def replace_node(genotype: Genotype, path: Path, new_node: Node) -> Genotype:
    """Return a deep copy of ``genotype`` whose body node at ``path`` is a copy of ``new_node``."""
    result = copy.deepcopy(genotype)
    new_node = copy.deepcopy(new_node)
    if not path:
        result["body"] = new_node
    else:
        get_node(result, path[:-1])["children"][path[-1]] = new_node
    return result
