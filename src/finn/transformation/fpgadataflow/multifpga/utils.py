"""Utilities for MultiFPGA Usage."""

import functools
from collections.abc import Callable
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any, ParamSpec, TypeVar

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import get_device_id

rT = TypeVar("rT")  # return type  # noqa: N816
pT = ParamSpec("pT")  # parameters type  # noqa: N816
argT = TypeVar("argT")  # type of the searched argument  # noqa: N816


def _find_argument(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any], cls: type[argT]
) -> argT:
    """Return the first positional or keyword argument passed to the function `name` that is an
    instance of `cls`. Raise a FINNInternalError if the function did not receive such an argument.
    """
    for arg in [*args, *kwargs.values()]:
        if isinstance(arg, cls):
            return arg
    raise FINNInternalError(
        f"{cls.__name__} argument required, but function does not receive "
        f"a {cls.__name__}: {name}"
    )


def ensure_multifpga_config(func: Callable[pT, rT]) -> Callable[pT, rT]:
    """Make sure that the passed function:
    1) Takes a DataflowBuildConfig as parameter
    2) The DataflowBuildConfig has a not-None MultiFPGA config field.
    """
    # Not every callable is a function with a __qualname__ (e.g. functools.partial)
    name = getattr(func, "__qualname__", repr(func))

    @functools.wraps(func)
    def wrapped(*args: pT.args, **kwargs: pT.kwargs) -> rT:
        """Check the MultiFPGA config and call the original function."""
        cfg = _find_argument(name, args, kwargs, DataflowBuildConfig)
        if cfg.multifpga_configuration is None:
            raise FINNInternalError(f"MultiFPGA config required but not found: {name}")
        return func(*args, **kwargs)

    return wrapped


def ensure_board_set(func: Callable[pT, rT]) -> Callable[pT, rT]:
    """Make sure that the passed function:
    1) Takes a DataflowBuildConfig as parameter
    2) The DataflowBuildConfig has a not-None board config field.
    """
    # Not every callable is a function with a __qualname__ (e.g. functools.partial)
    name = getattr(func, "__qualname__", repr(func))

    @functools.wraps(func)
    def wrapped(*args: pT.args, **kwargs: pT.kwargs) -> rT:
        """Check the config and call the original function."""
        cfg = _find_argument(name, args, kwargs, DataflowBuildConfig)
        if cfg.board is None:
            raise FINNInternalError(f"Board config parameter required but not found: {name}")
        return func(*args, **kwargs)

    return wrapped


def ensure_only_sdp_nodes(func: Callable[pT, rT]) -> Callable[pT, rT]:
    """Make sure that the passed function:
    1) Takes a ModelWrapper as an argument
    2) That every node in the modelwrapper is an SDP node.
    """
    # Not every callable is a function with a __qualname__ (e.g. functools.partial)
    name = getattr(func, "__qualname__", repr(func))

    @functools.wraps(func)
    def wrapped(*args: pT.args, **kwargs: pT.kwargs) -> rT:
        """Check that all nodes are SDP nodes and call the original function."""
        model = _find_argument(name, args, kwargs, ModelWrapper)
        for node in model.graph.node:
            if node.op_type != "StreamingDataflowPartition":
                raise FINNInternalError(
                    f"Function {name} requires all "
                    f"nodes to be StreamingDataflowPartition nodes. "
                    f"Node {node.name} is of type {node.op_type}!"
                )
        return func(*args, **kwargs)

    return wrapped


def ensure_device_id_set(func: Callable[pT, rT]) -> Callable[pT, rT]:
    """Make sure that the passed function:
    1) Takes a ModelWrapper as an argument
    2) Every node has a device ID.
    """
    # Not every callable is a function with a __qualname__ (e.g. functools.partial)
    name = getattr(func, "__qualname__", repr(func))

    @functools.wraps(func)
    def wrapped(*args: pT.args, **kwargs: pT.kwargs) -> rT:
        """Check that all nodes have a device ID and call the original function."""
        model = _find_argument(name, args, kwargs, ModelWrapper)
        for node in model.graph.node:
            if get_device_id(node) is None:
                raise FINNInternalError(
                    f"Function {name} requires all "
                    f"nodes to have a set Device ID. "
                    f"Node {node.name} does not."
                )
        return func(*args, **kwargs)

    return wrapped
