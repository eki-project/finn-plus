"""Guard against test-only dependencies leaking into the finn-plus package.

A plain ``pip install finn-plus`` does not install the test suite or its tooling (pytest
and friends, junitparser, ...; the ``test`` extra). Any module of the ``finn`` package
that imports one of them at module level breaks for those users, in the worst case
already on ``finn --help`` (see finn-plus 1.6.0, where ``finn.interface.manage_tests``
imported ``junitparser`` at module level). Such imports must stay local to the
function that needs them.
"""

import pytest

import ast
from collections.abc import Iterator
from pathlib import Path

import finn

# Top-level names that only the test suite (repository tests/ directory) or the "test"
# extra provide
TEST_ONLY_MODULES = frozenset(
    {
        "pytest",
        "_pytest",
        "xdist",
        "pytest_html",
        "pytest_timeout",
        "pytest_rerunfailures",
        "junitparser",
        "wget",
        "tests",
    }
)


def iter_module_level_imports(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield (line number, top-level module name) of every import executed on module import.

    Only statements executed unconditionally when the module is imported are considered:
    top-level imports and imports inside top-level ``try`` blocks. Imports inside functions,
    classes and ``if`` blocks (e.g. ``if TYPE_CHECKING:``) are fine, since they are not
    executed until the code is actually used.
    """
    statements = list(tree.body)
    for stmt in tree.body:
        if isinstance(stmt, ast.Try):
            statements.extend(stmt.body)
            statements.extend(stmt.finalbody)
    for stmt in statements:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                yield stmt.lineno, alias.name.split(".")[0]
        elif isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
            yield stmt.lineno, stmt.module.split(".")[0]


@pytest.mark.util
def test_no_test_only_imports_in_package() -> None:
    package_dir = Path(finn.__file__).parent
    offenders = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, module in iter_module_level_imports(tree):
            if module in TEST_ONLY_MODULES:
                offenders.append(f"{path.relative_to(package_dir)}:{lineno} imports {module}")
    assert offenders == [], (
        "Modules of the finn package import test-only dependencies at module level, which "
        "breaks a plain 'pip install finn-plus'. Move these imports into the functions that "
        "need them:\n" + "\n".join(offenders)
    )
