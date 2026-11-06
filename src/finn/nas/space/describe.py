"""Human-readable rendering of genotypes for logs."""

from __future__ import annotations

from finn.nas.space.genotype import Genotype, Node, genotype_hash, is_inner, iter_nodes
from finn.nas.space.shapes import Shape, infer_shapes


def _shape(shape: Shape) -> str:
    """Format a shape as ``CxHxW`` or ``F``."""
    return "x".join(str(dim) for dim in shape)


def _label(node: Node) -> str:
    """Return the one-line label of a node: kind plus compact hyperparameters."""
    kind = node["kind"]
    if kind == "branching":
        return f"branching {node['agg']}"
    if kind == "conv":
        channels = "depthwise" if node["depthwise"] else f"c{node['out_channels']}"
        return (
            f"conv k{node['kernel']} s{node['stride']} {node['padding']} {channels} "
            f"w{node['weight_bits']}"
        )
    if kind == "linear":
        return f"linear f{node['out_features']} w{node['weight_bits']}"
    if kind == "quant_act":
        return f"quant_act a{node['bits']}"
    if kind == "maxpool":
        return f"maxpool k{node['kernel']} s{node['stride']}"
    if kind == "avgpool":
        return f"avgpool k{node['kernel']}"
    return kind


def describe(genotype: Genotype) -> str:
    """Return an indented description with one line per node and its output shape.

    The shapes are recomputed here; they are not part of the genotype.
    """
    info = infer_shapes(genotype)
    head = genotype["head"]
    # A vector needs no pooling, so the head kind only matters after a feature map.
    reduce = head["kind"] if len(info.body_out) == 3 else "none"
    digest = genotype_hash(genotype)[:12]
    lines = [
        f"genotype {digest}: {info.num_nodes} nodes, depth {info.depth}, {info.params} params",
        f"input_quant a{genotype['input_bits']} -> {_shape(info.input_shape)}",
    ]
    for path, node in iter_nodes(genotype["body"]):
        indent = "  " * len(path)
        arrow = "=>" if is_inner(node) else "->"
        lines.append(f"{indent}{_label(node)} {arrow} {_shape(info.nodes[path].out_shape)}")
    lines.append(
        f"head {reduce} linear f{genotype['num_classes']} w{head['weight_bits']} "
        f"-> {_shape(info.output_shape)}"
    )
    return "\n".join(lines)
