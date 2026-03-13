"""Implements a decorator to mark functions as deprecated."""
import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

rT = TypeVar("rT")  # return type  # noqa: N816
pT = ParamSpec("pT")  # parameters type  # noqa: N816


def deprecated(func: Callable[pT, rT]) -> Callable[pT, rT]:
    """Use this decorator to mark functions as deprecated.
    Every time the decorated function runs, it will emit
    a "deprecation" warning.
    """

    @functools.wraps(func)
    def new_func(*args: pT.args, **kwargs: pT.kwargs) -> rT:
        log.warning(
            f"Using {func.__qualname__} is deprecated and will be removed in the next release.",
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func
