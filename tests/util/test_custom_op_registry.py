"""Tests for the fpgadataflow custom-op registry and its qonnx integration."""

import pytest

import sys
import types
from onnx import NodeProto, helper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import get_ops_in_domain, getCustomOp

import finn.custom_op.fpgadataflow as abstract_ops
import finn.custom_op.fpgadataflow.hls as hls_ops
import finn.custom_op.fpgadataflow.rtl as rtl_ops
from finn.custom_op.fpgadataflow._registry import make_registry


def _make_dummy_op(name: str) -> type[CustomOp]:
    """Return a minimal concrete CustomOp subclass called ``name``."""

    def get_nodeattr_types(self: CustomOp) -> dict:  # noqa: ARG001
        return {}

    def unused(self: CustomOp, *args: object) -> None:  # noqa: ARG001
        return None

    def verify_node(self: CustomOp) -> list[str]:  # noqa: ARG001
        return []

    return type(
        name,
        (CustomOp,),
        {
            "get_nodeattr_types": get_nodeattr_types,
            "make_shape_compatible_op": unused,
            "infer_node_datatype": unused,
            "execute_node": unused,
            "verify_node": verify_node,
        },
    )


def _node(domain: str, op_type: str) -> NodeProto:
    """Return a single-input, single-output node of the given domain and type."""
    return helper.make_node(op_type, ["in0"], ["out0"], domain=domain)


@pytest.mark.util
@pytest.mark.parametrize("package", [abstract_ops, hls_ops, rtl_ops], ids=lambda p: p.__name__)
def test_domain_lists_exactly_the_registered_ops(package: types.ModuleType) -> None:
    """get_ops_in_domain lists every registered op and nothing else (no base classes)."""
    listed = {name for name, _ in get_ops_in_domain(package.__name__)}
    assert listed == set(package.custom_op)


@pytest.mark.util
def test_op_registered_after_import_is_listed_and_resolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An op registered after its domain package was imported is listed and resolves."""
    domain = "finn_test_registry.late_registration"
    package = types.ModuleType(domain)
    custom_op, register = make_registry(CustomOp, domain)
    monkeypatch.setattr(package, "custom_op", custom_op, raising=False)
    monkeypatch.setitem(sys.modules, domain, package)
    # The package is imported (and get_ops_in_domain was called) before the op exists
    assert get_ops_in_domain(domain) == []

    late_op = register(_make_dummy_op("LateOp"))

    assert ("LateOp", late_op) in get_ops_in_domain(domain)
    assert isinstance(getCustomOp(_node(domain, "LateOp")), late_op)


@pytest.mark.util
def test_reregistering_an_op_replaces_the_resolved_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registering a class under an already resolved name takes effect immediately."""
    domain = "finn_test_registry.reregistration"
    package = types.ModuleType(domain)
    custom_op, register = make_registry(CustomOp, domain)
    monkeypatch.setattr(package, "custom_op", custom_op, raising=False)
    monkeypatch.setitem(sys.modules, domain, package)

    first = register(_make_dummy_op("SomeOp"))
    assert isinstance(getCustomOp(_node(domain, "SomeOp")), first)
    second = register(_make_dummy_op("SomeOp"))

    assert type(getCustomOp(_node(domain, "SomeOp"))) is second
    assert dict(get_ops_in_domain(domain))["SomeOp"] is second
