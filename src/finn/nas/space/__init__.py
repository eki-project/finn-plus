"""Grammar-based NAS search space (einspace-style) over layers FINN can in principle map.

Genotypes are plain JSON data; sampling, validation and variation need neither torch nor
FINN. Only :mod:`finn.nas.space.build` (``to_module``) imports torch and Brevitas, so it is
not imported here: use ``from finn.nas.space.build import to_module``.
"""

from finn.nas.space.config import SpaceConfig
from finn.nas.space.describe import describe
from finn.nas.space.genotype import Genotype, InvalidGenotypeError, canonical_json, genotype_hash
from finn.nas.space.sampler import SamplingError
from finn.nas.space.search_space import SearchSpace
from finn.nas.space.seeds import SEEDS, seed_genotypes
from finn.nas.space.shapes import GenotypeInfo, count_params, infer_shapes, validate

__all__ = [
    "SEEDS",
    "Genotype",
    "GenotypeInfo",
    "InvalidGenotypeError",
    "SamplingError",
    "SearchSpace",
    "SpaceConfig",
    "canonical_json",
    "count_params",
    "describe",
    "genotype_hash",
    "infer_shapes",
    "seed_genotypes",
    "validate",
]
