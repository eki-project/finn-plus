"""Declarative parameter spaces for microbenchmark DUTs.

A :data:`ParamSpace` maps parameter names to :class:`Dimension` objects in sampling order.
Dimensions are sampled one after another and may depend on the values sampled so far
(``partial``), e.g. a folding factor that must divide a previously sampled matrix width.
The sampler in :mod:`finn.benchmarking.sampling` draws candidates from a space and hands
them to the DUT's ``validate`` function, which encodes the remaining cross-parameter
constraints.

Pure python (random/math only) so that it can be used and tested without FINN.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Dimension(Protocol):
    """One sampled parameter."""

    def sample(self, rng: random.Random, partial: dict) -> Any:
        """Draw a value; ``partial`` holds the parameters sampled so far."""
        ...


def _weighted_choice(rng: random.Random, values: Sequence, weights: Optional[Sequence[float]]):
    if weights is None:
        return rng.choice(list(values))
    return rng.choices(list(values), weights=list(weights), k=1)[0]


@dataclass(frozen=True)
class Choice:
    """Uniform (or weighted) choice from a fixed list of values."""

    values: list
    weights: Optional[list[float]] = None

    def sample(self, rng: random.Random, partial: dict) -> Any:
        return _weighted_choice(rng, self.values, self.weights)


@dataclass(frozen=True)
class IntRange:
    """Integer in ``[lo, hi]``; ``log=True`` samples log-uniformly; ``step`` rounds to a grid."""

    lo: int
    hi: int
    log: bool = False
    step: int = 1

    def sample(self, rng: random.Random, partial: dict) -> int:
        if self.log:
            value = int(round(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))))
        else:
            value = rng.randint(self.lo, self.hi)
        value = max(self.lo, min(self.hi, value))
        if self.step > 1:
            value = max(self.lo, (value // self.step) * self.step)
        return value


@dataclass(frozen=True)
class Pow2Range:
    """Power of two in ``[lo, hi]`` (log-uniform, i.e. uniform over the exponents)."""

    lo: int
    hi: int
    weights: Optional[list[float]] = None

    def values(self) -> list[int]:
        e_lo = math.ceil(math.log2(self.lo))
        e_hi = math.floor(math.log2(self.hi))
        return [2**e for e in range(e_lo, e_hi + 1)]

    def sample(self, rng: random.Random, partial: dict) -> int:
        return _weighted_choice(rng, self.values(), self.weights)


@dataclass(frozen=True)
class FloatRange:
    """Float in ``[lo, hi]``; ``log=True`` samples log-uniformly; ``step`` rounds to a grid."""

    lo: float
    hi: float
    log: bool = False
    step: Optional[float] = None

    def sample(self, rng: random.Random, partial: dict) -> float:
        if self.log:
            value = math.exp(rng.uniform(math.log(self.lo), math.log(self.hi)))
        else:
            value = rng.uniform(self.lo, self.hi)
        if self.step:
            value = round(round(value / self.step) * self.step, 10)
        return max(self.lo, min(self.hi, value))


def divisors(n: int, pow2: bool = False) -> list[int]:
    """All divisors of ``n`` (or only its power-of-two divisors)."""
    if pow2:
        result = []
        d = 1
        while n % d == 0:
            result.append(d)
            d *= 2
        return result
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True)
class Divisor:
    """Uniform choice among the divisors of an earlier parameter ``of`` (e.g. SIMD dividing
    the matrix width). ``extra`` values (e.g. ``-1`` for "maximum") are added with the given
    total weight."""

    of: str
    pow2: bool = True
    extra: tuple = ()
    extra_weight: float = 0.0
    lo: int = 1
    hi: Optional[int] = None

    def sample(self, rng: random.Random, partial: dict) -> int:
        n = int(partial[self.of])
        candidates = [d for d in divisors(n, self.pow2) if d >= self.lo]
        if self.hi is not None:
            candidates = [d for d in candidates if d <= self.hi]
        if not candidates:
            candidates = [1]
        if self.extra and self.extra_weight > 0 and rng.random() < self.extra_weight:
            return rng.choice(list(self.extra))
        return rng.choice(candidates)


@dataclass(frozen=True)
class Derived:
    """Deterministic function of the parameters sampled so far."""

    fn: Callable[[dict], Any]

    def sample(self, rng: random.Random, partial: dict) -> Any:
        return self.fn(partial)


@dataclass(frozen=True)
class Fixed:
    """A constant value (kept in the space so that it is part of the parameter key)."""

    value: Any

    def sample(self, rng: random.Random, partial: dict) -> Any:
        return self.value


#: Parameter name -> dimension, in sampling order (conditional dimensions after their deps)
ParamSpace = dict[str, Dimension]


@dataclass
class SpaceOverride:
    """Overrides applied to a DUT's default space (see ``finn.benchmarking.sampling``)."""

    dims: dict[str, Dimension] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)


def sample_params(space: ParamSpace, rng: random.Random, fixed: Optional[dict] = None) -> dict:
    """Draw one parameter set: fixed values first, then every dimension in space order."""
    params: dict = dict(fixed or {})
    for name, dim in space.items():
        if name in params:
            continue
        params[name] = dim.sample(rng, params)
    return params


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


@dataclass(frozen=True)
class Conditional:
    """Select a dimension depending on the value of an earlier parameter ``on``:
    ``cases[key(value)]`` if present, else ``default``. Lists are looked up as tuples."""

    on: str
    cases: dict
    default: Any
    key: Optional[Callable[[Any], Any]] = None

    def sample(self, rng: random.Random, partial: dict) -> Any:
        value = partial.get(self.on)
        if self.key is not None:
            value = self.key(value)
        dim = self.cases.get(_hashable(value), self.default)
        return dim.sample(rng, partial)
