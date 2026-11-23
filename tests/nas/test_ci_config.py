"""The search space config shipped for the CI (ci/nas/default.json) matches the code's defaults."""

import pytest

import json
from pathlib import Path

space = pytest.importorskip("finn.nas.space")

CI_CONFIG = Path(__file__).resolve().parents[2] / "ci" / "nas" / "default.json"


@pytest.mark.util
def test_default_ci_space_config_matches_code_defaults() -> None:
    """A changed default in SpaceConfig must be reflected in the CI config (and vice versa)."""
    if not CI_CONFIG.exists():
        pytest.skip(f"{CI_CONFIG} not available (tests run outside the repository checkout)")
    data = json.loads(CI_CONFIG.read_text())
    assert space.SpaceConfig.from_dict(data) == space.SpaceConfig()
    assert data == space.SpaceConfig().to_dict()
