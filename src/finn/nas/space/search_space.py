"""The search space object a NAS algorithm talks to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.nas.space import variation
from finn.nas.space.config import SpaceConfig
from finn.nas.space.sampler import Sampler
from finn.nas.space.shapes import GenotypeInfo, validate

if TYPE_CHECKING:
    import random

    from finn.nas.space.genotype import Genotype


class SearchSpace:
    """Sampling and variation over the genotypes of one :class:`SpaceConfig`.

    All randomness comes from the ``random.Random`` passed to each call, so results are
    reproducible and independent of global random state::

        space = SearchSpace()
        rng = random.Random(0)
        parent = space.sample(rng)
        child = space.mutate(parent, rng)
        net = to_module(child)  # finn.nas.space.build, needs torch + brevitas
    """

    def __init__(self, config: SpaceConfig | None = None) -> None:
        """Create the space; ``config`` defaults to small CIFAR-10 networks."""
        self.config = config if config is not None else SpaceConfig()
        self._sampler = Sampler(self.config)

    def sample(self, rng: random.Random) -> Genotype:
        """Draw a random genotype; raises ``SamplingError`` if none is found."""
        return self._sampler.sample(rng)

    def mutate(self, genotype: Genotype, rng: random.Random) -> Genotype:
        """Return a valid mutated copy of ``genotype`` (see :func:`variation.mutate`)."""
        return variation.mutate(genotype, rng, self._sampler)

    def crossover(self, a: Genotype, b: Genotype, rng: random.Random) -> tuple[Genotype, Genotype]:
        """Return two valid children of ``a`` and ``b`` (see :func:`variation.crossover`)."""
        return variation.crossover(a, b, rng, self._sampler)

    def validate(self, genotype: Genotype) -> GenotypeInfo:
        """Check membership of ``genotype`` in this space and return its shape information."""
        return validate(genotype, self.config)
