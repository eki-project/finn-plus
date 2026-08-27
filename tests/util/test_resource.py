from finn.util.resources import _merge_resource_estimations

# TODO: Test: modelwrapper method, available resources, etc.


def test_resource_merge() -> None:
    """Test that the resources estimates are being merged correctly."""
    a = {"A": {"LUT": 100, "BRAM": 10}, "B": {"LUT": 120, "BRAM": 2}, "D": {"LUT": 500}}
    b = {
        "A": {
            "LUT": 100,
            "DSP": 12,
        },
        "B": {"LUT": 120, "BRAM": 5},
        "C": {"LUT": 90},
    }
    merged = _merge_resource_estimations(a, b)  # type:ignore

    # All nodes' resources were merged
    for node in ["A", "B", "C", "D"]:
        assert node in merged

    # Test that the layer that only exists in one estimate is used
    assert merged["C"]["LUT"] == 90
    assert merged["D"]["LUT"] == 500

    # Test that the larger value is used if available
    assert merged["B"]["BRAM"] == 5

    # Test that missing resource types are filled by the other estimate
    assert merged["A"]["BRAM"] == 10
    assert merged["A"]["DSP"] == 12
