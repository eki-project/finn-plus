"""Tests for the node-carrying exception types and the errors of utility functions that use
FINN+'s exception hierarchy."""

import pytest

import pickle

from finn.util.basic import get_vivado_root
from finn.util.exception import (
    FINNConfigurationError,
    FINNError,
    FINNFoldingError,
    FINNNodeError,
    FINNUnsupportedOpError,
    FINNUserError,
)


@pytest.mark.util
@pytest.mark.parametrize("error_type", [FINNNodeError, FINNUnsupportedOpError, FINNFoldingError])
def test_node_errors_carry_the_node(error_type: type[FINNNodeError]) -> None:
    """Node errors are user errors that carry an optional node name and op type."""
    err = error_type("node_0: something is wrong", node_name="node_0", op_type="MVAU_hls")
    assert isinstance(err, FINNNodeError)
    assert isinstance(err, FINNUserError)
    assert isinstance(err, FINNError)
    assert str(err) == err.msg == "node_0: something is wrong"
    assert (err.node_name, err.op_type) == ("node_0", "MVAU_hls")
    # the node is optional
    err = error_type("something is wrong")
    assert (err.node_name, err.op_type) == (None, None)


@pytest.mark.util
def test_node_error_details() -> None:
    """The additional details of the node errors default to empty containers."""
    assert FINNUnsupportedOpError("msg").unmapped_nodes == []
    assert FINNFoldingError("msg").violations == {}
    unmapped = [{"name": "a", "op_type": "Mul", "domain": "", "position": "interior"}]
    assert FINNUnsupportedOpError("msg", "a", "Mul", unmapped_nodes=unmapped).unmapped_nodes == (
        unmapped
    )
    violations = {"a": "a: illegal"}
    assert FINNFoldingError("msg", "a", "MVAU_hls", violations=violations).violations == violations


@pytest.mark.util
def test_node_errors_can_be_pickled() -> None:
    """Node errors survive pickling, e.g. when handed from a worker process to its parent."""
    unmapped = [{"name": "a", "op_type": "Mul", "domain": "", "position": "interior"}]
    for err in (
        FINNUnsupportedOpError("msg", "a", "Mul", unmapped_nodes=unmapped),
        FINNFoldingError("msg", "a", "MVAU_hls", violations={"a": "a: illegal"}),
    ):
        clone = pickle.loads(pickle.dumps(err))
        assert type(clone) is type(err)
        assert str(clone) == "msg"
        assert vars(clone) == vars(err)


@pytest.mark.util
def test_get_vivado_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_vivado_root raises a configuration error if XILINX_VIVADO is not set."""
    monkeypatch.setenv("XILINX_VIVADO", "/opt/Xilinx/Vivado/2024.2")
    assert get_vivado_root() == "/opt/Xilinx/Vivado/2024.2"
    monkeypatch.delenv("XILINX_VIVADO")
    with pytest.raises(FINNConfigurationError, match="XILINX_VIVADO") as excinfo:
        get_vivado_root()
    assert isinstance(excinfo.value, FINNUserError)
