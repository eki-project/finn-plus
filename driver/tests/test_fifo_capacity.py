"""Tests of the driver's mirror of the ``fifo.sv`` capacity (``fifo_capacity``).

The finn test suite checks the mirror against ``finn.util.fifo_cost.geometry``; here the
ladder is checked against the RTL facts directly and the plateau tops against a full scan.
"""

import pytest

from finn_plus_driver.overlays.fifo_capacity import (
    STYLES,
    CapacityModel,
    capacity,
    identity_above_shift,
    plateau_top,
    plateau_tops,
    resolve_style,
)


def test_capacity_ladder_of_fifo_sv():
    """Shift floor, LUTRAM identity, BRAM and URAM rounding at the thresholds of fifo.sv."""
    assert [capacity(d, 32) for d in (2, 3, 4, 5, 6, 33)] == [5, 5, 5, 5, 6, 33]
    assert resolve_style(33, 32) == "srl" and resolve_style(34, 32) == "distributed"
    assert resolve_style(64, 8) == "srl" and resolve_style(65, 8) == "distributed"
    assert resolve_style(1000, 4) == "distributed" and capacity(1000, 4) == 1000
    assert resolve_style(1000, 5) == "block" and resolve_style(2029, 5) == "ultra"
    assert capacity(257, 32) == 257 and capacity(258, 32) == 515 and capacity(513, 32) == 515
    assert capacity(514, 32) == 517 and capacity(769, 32) == 771 and capacity(1025, 32) == 1027
    assert capacity(2028, 32) == 2051 and capacity(2029, 32) == 2068 and capacity(3000, 32) == 4116
    assert capacity(49830, 8) == 65563
    # a requested style is honoured above the shift floor only
    assert resolve_style(20, 32, "ultra") == "srl" and capacity(20, 32, "ultra") == 20
    assert resolve_style(100, 32, "block") == "block" and capacity(100, 32, "block") == 131
    assert resolve_style(5000, 32, "srl") == "srl" and capacity(5000, 32, "srl") == 5000
    with pytest.raises(ValueError):
        capacity(1, 32)
    with pytest.raises(ValueError):
        resolve_style(10, 32, "lutram")


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("width", [1, 4, 8, 12, 32, 64])
def test_plateau_tops_match_a_full_scan(style, width):
    """The analytic plateau tops are exactly the depths after which the capacity changes."""
    max_depth = 6000
    caps = [capacity(d, width, style) for d in range(2, max_depth + 2)]
    scan = [d for d in range(2, max_depth + 1) if d == max_depth or caps[d - 1] != caps[d - 2]]
    assert plateau_tops(width, max_depth, style) == scan
    for d in range(2, 2500, 7):
        top = plateau_top(d, width, style)
        assert top >= d and capacity(top, width, style) == capacity(d, width, style)
        assert capacity(top + 1, width, style) != capacity(d, width, style)


def test_identity_and_capacity_model():
    """Narrow or forced LUTRAM/SRL FIFOs have no plateaus above the floor; the model indexes."""
    assert identity_above_shift(4) and not identity_above_shift(5)
    assert identity_above_shift(32, "distributed") and identity_above_shift(32, "srl")
    assert not identity_above_shift(32, "block")
    model = CapacityModel([8, 32], ["auto", "block"])
    assert model.num_fifos == 2
    # a forced block RAM: 64 rows for depths 34..65, 128 rows for 66..129, ...
    assert model.capacity(0, 2) == 5 and model.capacity(1, 100) == 131
    assert model.plateau_top(0, 3) == 5 and model.plateau_top(1, 100) == 129
    assert model.plateau_tops(1, 300) == [5] + list(range(6, 34)) + [65, 129, 257, 300]
    assert model.identity_above_shift(0) is False and model.identity_above_shift(1) is False
    with pytest.raises(ValueError):
        CapacityModel([8, 32], ["auto"])
