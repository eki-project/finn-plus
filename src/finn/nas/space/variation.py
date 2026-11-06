"""Variation operators: subtree/hyperparameter mutation and subtree crossover.

All operators return new genotypes, never touch their inputs, and only return genotypes that
validate against the config. As in the sampler, validity means "PyTorch can run it and it is
within budget"; there is no FINN knowledge here.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from finn.nas.space.genotype import (
    Genotype,
    InvalidGenotypeError,
    Path,
    genotype_hash,
    get_node,
    is_inner,
    iter_nodes,
    replace_node,
)
from finn.nas.space.sampler import DeadEndError, Sampler
from finn.nas.space.shapes import GenotypeInfo, validate

if TYPE_CHECKING:
    import random

    from finn.nas.space.config import SpaceConfig

#: Pseudo path of the genes outside the body (input quantizer and head).
_WRAPPER = None


def _resample_subtree(
    genotype: Genotype, info: GenotypeInfo, rng: random.Random, sampler: Sampler
) -> Genotype | None:
    """Replace a random subtree by a fresh sample for the same input shape (einspace-style).

    The replacement may use the budget the old subtree occupied. Its output shape is free;
    whether the rest of the network still fits is decided by the caller's validation.
    """
    cfg = sampler.config
    path = rng.choice(list(info.nodes))
    old = info.nodes[path]
    try:
        if not path:
            new = sampler.body(rng)
        else:
            new = sampler.module(
                rng,
                old.in_shape,
                len(path),
                cfg.max_nodes - (info.num_nodes - old.size),
                cfg.max_params - (info.params - old.params),
            )
    except DeadEndError:
        return None
    return replace_node(genotype, path, new.node)


def _mutable_genes(genotype: Genotype, config: SpaceConfig) -> dict[Path | None, dict[str, list]]:
    """Return, per terminal (and for the wrapper), the genes that have an alternative value."""
    current = {
        "input_bits": genotype["input_bits"],
        "head.kind": genotype["head"]["kind"],
        "head.weight_bits": genotype["head"]["weight_bits"],
    }
    genes: dict[Path | None, dict[str, list]] = {
        _WRAPPER: {
            name: [v for v in values if v != current[name]]
            for name, values in config.wrapper_choices().items()
        }
    }
    for path, node in iter_nodes(genotype["body"]):
        if is_inner(node):
            continue
        genes[path] = {
            name: [v for v in values if v != node[name]]
            for name, values in config.choices(node["kind"]).items()
            if name in node
        }
    pruned = {
        target: {name: values for name, values in names.items() if values}
        for target, names in genes.items()
    }
    return {target: names for target, names in pruned.items() if names}


def _set_gene(
    genotype: Genotype,
    target: Path | None,
    name: str,
    value: Any,
    rng: random.Random,
    config: SpaceConfig,
) -> None:
    """Assign one gene in place, keeping the node's field set consistent."""
    if target is _WRAPPER:
        if name == "input_bits":
            genotype["input_bits"] = value
        else:
            genotype["head"][name.removeprefix("head.")] = value
        return
    node = get_node(genotype, target)
    node[name] = value
    if name == "depthwise":
        # out_channels only exists for dense convolutions.
        if value:
            del node["out_channels"]
        else:
            node["out_channels"] = rng.choice(config.conv_out_channels)


def _perturb_gene(genotype: Genotype, rng: random.Random, config: SpaceConfig) -> Genotype | None:
    """Give one hyperparameter of a random terminal (or of the wrapper) another value."""
    genes = _mutable_genes(genotype, config)
    if not genes:
        return None
    target = rng.choice(list(genes))
    name = rng.choice(list(genes[target]))
    child = copy.deepcopy(genotype)
    _set_gene(child, target, name, rng.choice(genes[target][name]), rng, config)
    return child


def mutate(genotype: Genotype, rng: random.Random, sampler: Sampler) -> Genotype:
    """Return a mutated copy of ``genotype`` that validates against the sampler's config.

    Each attempt either resamples a random subtree or perturbs one hyperparameter. Attempts
    that break the network (e.g. a changed stride under an add) or reproduce the parent are
    discarded. If all ``max_variation_attempts`` fail, a copy of the parent is returned.
    """
    cfg = sampler.config
    info = validate(genotype, cfg)
    parent_hash = genotype_hash(genotype)
    for _ in range(cfg.max_variation_attempts):
        if rng.random() < cfg.subtree_mutation_prob:
            child = _resample_subtree(genotype, info, rng, sampler)
        else:
            child = _perturb_gene(genotype, rng, cfg)
        if child is None or genotype_hash(child) == parent_hash:
            continue
        try:
            validate(child, cfg)
        except InvalidGenotypeError:
            continue
        return child
    return copy.deepcopy(genotype)


def crossover(
    a: Genotype, b: Genotype, rng: random.Random, sampler: Sampler
) -> tuple[Genotype, Genotype]:
    """Exchange a random subtree of ``a`` with a random subtree of ``b``.

    Returns the two children (``a`` with the subtree of ``b`` and vice versa). Pairs of
    subtrees are drawn until both children validate and are new; if that does not happen
    within ``max_variation_attempts``, the children are mutations of the parents instead.
    """
    cfg = sampler.config
    paths_a = list(validate(a, cfg).nodes)
    paths_b = list(validate(b, cfg).nodes)
    parents = {genotype_hash(a), genotype_hash(b)}
    for _ in range(cfg.max_variation_attempts):
        path_a, path_b = rng.choice(paths_a), rng.choice(paths_b)
        child_a = replace_node(a, path_a, get_node(b, path_b))
        child_b = replace_node(b, path_b, get_node(a, path_a))
        if genotype_hash(child_a) in parents or genotype_hash(child_b) in parents:
            continue
        try:
            validate(child_a, cfg)
            validate(child_b, cfg)
        except InvalidGenotypeError:
            continue
        return child_a, child_b
    return mutate(a, rng, sampler), mutate(b, rng, sampler)
