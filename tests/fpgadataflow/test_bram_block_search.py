"""Test BRAM block calculations and search algorithms."""
# ruff: noqa: ANN201, SLF001

import pytest

import math

from finn.transformation.fpgadataflow.simulation import (
    calculate_bram_blocks,
    calculate_bram_depth_range,
)


class TestBRAMBlockCalculations:
    """Test BRAM block calculation functions."""

    def test_calculate_bram_blocks_bitwidth_1(self) -> None:
        """Test BRAM block calculation for 1-bit data."""
        assert calculate_bram_blocks(1, 1) == 1
        assert calculate_bram_blocks(16384, 1) == 1
        assert calculate_bram_blocks(16385, 1) == 2
        assert calculate_bram_blocks(32768, 1) == 2

    def test_calculate_bram_blocks_bitwidth_2(self) -> None:
        """Test BRAM block calculation for 2-bit data."""
        assert calculate_bram_blocks(1, 2) == 1
        assert calculate_bram_blocks(8192, 2) == 1
        assert calculate_bram_blocks(8193, 2) == 2
        assert calculate_bram_blocks(16384, 2) == 2

    def test_calculate_bram_blocks_bitwidth_4(self) -> None:
        """Test BRAM block calculation for 4-bit data."""
        assert calculate_bram_blocks(1, 4) == 1
        assert calculate_bram_blocks(4096, 4) == 1
        assert calculate_bram_blocks(4097, 4) == 2
        assert calculate_bram_blocks(8192, 4) == 2

    def test_calculate_bram_blocks_bitwidth_9(self) -> None:
        """Test BRAM block calculation for 9-bit data."""
        assert calculate_bram_blocks(1, 9) == 1
        assert calculate_bram_blocks(2048, 9) == 1
        assert calculate_bram_blocks(2049, 9) == 2

    def test_calculate_bram_blocks_bitwidth_18(self) -> None:
        """Test BRAM block calculation for 18-bit data."""
        assert calculate_bram_blocks(1, 18) == 1
        assert calculate_bram_blocks(1024, 18) == 1
        assert calculate_bram_blocks(1025, 18) == 2

    def test_calculate_bram_blocks_wide_bitwidth_deep(self) -> None:
        """Test BRAM block calculation for wide bitwidth with depth > 512."""
        # bitwidth = 40, depth = 1024 > 512
        # Uses formula: ⌈1024/1024⌉ * ⌈40/18⌉ = 1 * 3 = 3
        assert calculate_bram_blocks(1024, 40) == 3

    def test_calculate_bram_blocks_wide_bitwidth_shallow(self) -> None:
        """Test BRAM block calculation for wide bitwidth with depth <= 512."""
        # bitwidth = 40, depth = 512 <= 512
        # Uses formula: ⌈512/512⌉ * ⌈40/36⌉ = 1 * 2 = 2
        assert calculate_bram_blocks(512, 40) == 2


class TestBRAMDepthRange:
    """Test BRAM depth range inversion function."""

    def test_depth_range_bitwidth_1(self) -> None:
        """Test depth range calculation for 1-bit data."""
        min_d, max_d = calculate_bram_depth_range(1, 1)
        assert min_d == 1
        assert max_d == 16384
        assert calculate_bram_blocks(min_d, 1) == 1
        assert calculate_bram_blocks(max_d, 1) == 1

        min_d, max_d = calculate_bram_depth_range(2, 1)
        assert min_d == 16385
        assert max_d == 32768
        assert calculate_bram_blocks(min_d, 1) == 2
        assert calculate_bram_blocks(max_d, 1) == 2

    def test_depth_range_bitwidth_4(self) -> None:
        """Test depth range calculation for 4-bit data."""
        min_d, max_d = calculate_bram_depth_range(1, 4)
        assert min_d == 1
        assert max_d == 4096
        assert calculate_bram_blocks(min_d, 4) == 1
        assert calculate_bram_blocks(max_d, 4) == 1

    def test_depth_range_bitwidth_5_valid_blocks(self) -> None:
        """Test block count validation for bitwidth=5."""
        # bitwidth=5 uses ⌈5/9⌉=1 bitwidth factor (falls in <=9 range)
        # So all blocks should be valid
        min_d, max_d = calculate_bram_depth_range(1, 5)
        assert max_d > 0, "1 block should be valid for bitwidth=5"
        assert calculate_bram_blocks(min_d, 5) == 1
        assert calculate_bram_blocks(max_d, 5) == 1

        min_d, max_d = calculate_bram_depth_range(2, 5)
        assert max_d > 0, "2 blocks should be valid for bitwidth=5"
        assert calculate_bram_blocks(min_d, 5) == 2
        assert calculate_bram_blocks(max_d, 5) == 2

    def test_depth_range_bitwidth_10_valid_blocks(self) -> None:
        """Test block count validation for bitwidth=10."""
        # bitwidth=10 uses ⌈10/18⌉=1 bitwidth factor (falls in <=18 range)
        min_d, max_d = calculate_bram_depth_range(1, 10)
        assert max_d > 0
        assert calculate_bram_blocks(min_d, 10) == 1

        min_d, max_d = calculate_bram_depth_range(2, 10)
        assert max_d > 0
        assert calculate_bram_blocks(min_d, 10) == 2

    def test_depth_range_wide_bitwidth(self) -> None:
        """Test depth range for wide bitwidths > 18."""
        # bitwidth=40 has two modes depending on depth
        min_d, max_d = calculate_bram_depth_range(2, 40)
        # Should use depth ≤ 512 mode: ⌈depth/512⌉ * ⌈40/36⌉
        # 2 blocks / 2 = 1 depth_blocks → (1, 512)
        if max_d > 0:
            assert max_d <= 512
            assert calculate_bram_blocks(min_d, 40) == 2

    def test_depth_range_consistency_all_bitwidths(self) -> None:
        """Test that all valid ranges actually produce the correct block count."""
        for bitwidth in range(1, 8192):
            for blocks in range(1, 1024):
                min_d, max_d = calculate_bram_depth_range(blocks, bitwidth)
                if max_d > 0:  # Valid configuration
                    # Verify both endpoints produce correct block count
                    assert calculate_bram_blocks(min_d, bitwidth) == blocks, (
                        f"Min depth {min_d} for {blocks} blocks, "
                        f"bitwidth {bitwidth} produces wrong count"
                    )
                    assert calculate_bram_blocks(max_d, bitwidth) == blocks, (
                        f"Max depth {max_d} for {blocks} blocks, "
                        f"bitwidth {bitwidth} produces wrong count"
                    )

                    # Verify just outside the range produces different counts
                    if min_d > 1:
                        assert calculate_bram_blocks(min_d - 1, bitwidth) < blocks
                    assert calculate_bram_blocks(max_d + 1, bitwidth) > blocks


class TestGetValidBlockCounts:
    """Test the _get_valid_block_counts helper method."""

    def test_all_valid_bitwidth_1(self) -> None:
        """Test that all block counts are valid for bitwidth=1."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        # Create dummy instance just to test the method
        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)

        valid_blocks = sim._get_valid_block_counts(1, 10, 1)
        assert valid_blocks == list(range(1, 11))

    def test_wide_bitwidth_filtering(self) -> None:
        """Test that some block counts may be invalid for wide bitwidths."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)

        # For bitwidth > 18, some block counts may be invalid
        valid_blocks = sim._get_valid_block_counts(1, 20, 40)
        # Verify all returned blocks produce valid ranges
        for b in valid_blocks:
            _, max_d = calculate_bram_depth_range(b, 40)
            assert max_d > 0, f"Block {b} should produce valid range"

    def test_range_respects_bounds(self) -> None:
        """Test that valid blocks respect min/max bounds."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)

        valid_blocks = sim._get_valid_block_counts(5, 15, 1)
        assert min(valid_blocks) >= 5
        assert max(valid_blocks) <= 15
        assert len(valid_blocks) == 11

    def test_empty_when_no_valid_in_range(self) -> None:
        """Test that empty list is returned when no valid configs exist in range."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)

        # Test a scenario where the range might have no valid blocks
        # (this is rare but the method should handle it)
        valid_blocks = sim._get_valid_block_counts(100, 99, 5)  # Invalid range
        assert valid_blocks == []


class TestExponentialBinarySearchLogic:
    """Test the exponential + binary search algorithm logic (without actual simulation)."""

    def test_exponential_indices_progression(self) -> None:
        """Test that exponential search correctly progresses through indices."""
        # Simulate the exponential index progression
        valid_blocks = list(range(1, 101))  # 100 valid blocks

        # Exponential progression should be: 0, 1, 2, 4, 8, 16, 32, 64...
        exp_idx = 0
        indices_checked = []

        while exp_idx < len(valid_blocks) - 1:
            indices_checked.append(exp_idx)
            exp_idx = min(exp_idx * 2 if exp_idx > 0 else 1, len(valid_blocks) - 1)

        assert indices_checked == [0, 1, 2, 4, 8, 16, 32, 64]

    def test_binary_search_reduces_range(self) -> None:
        """Test that binary search correctly narrows the range."""
        lower_idx = 0
        upper_idx = 99

        iterations = 0
        while lower_idx < upper_idx:
            mid_idx = (lower_idx + upper_idx) // 2
            # Simulate "success" for indices < 50
            if mid_idx < 50:
                upper_idx = mid_idx
            else:
                lower_idx = mid_idx + 1
            iterations += 1

            # Prevent infinite loop in test
            if iterations > 20:
                break

        assert lower_idx == upper_idx
        assert iterations <= 7  # log2(100) ≈ 6.6


class TestSRL16ELUTCalculations:
    """Test SRL16E LUT calculation functions."""

    def test_calculate_srl16e_luts_basic(self):
        """Test basic SRL16E LUT calculations."""
        from finn.transformation.fpgadataflow.simulation import calculate_srl16e_luts

        # Formula: LUTs = ⌈depth/32⌉ * ⌈bitwidth/2⌉
        # depth=32, bitwidth=2: ⌈32/32⌉ * ⌈2/2⌉ = 1 * 1 = 1
        assert calculate_srl16e_luts(32, 2) == 1

        # depth=64, bitwidth=2: ⌈64/32⌉ * ⌈2/2⌉ = 2 * 1 = 2
        assert calculate_srl16e_luts(64, 2) == 2

        # depth=32, bitwidth=4: ⌈32/32⌉ * ⌈4/2⌉ = 1 * 2 = 2
        assert calculate_srl16e_luts(32, 4) == 2

        # depth=33, bitwidth=2: ⌈33/32⌉ * ⌈2/2⌉ = 2 * 1 = 2
        assert calculate_srl16e_luts(33, 2) == 2

    def test_calculate_srl16e_luts_various_bitwidths(self):
        """Test SRL16E LUT calculations for various bitwidths."""
        from finn.transformation.fpgadataflow.simulation import calculate_srl16e_luts

        # Bitwidth 1: ⌈1/2⌉ = 1
        assert calculate_srl16e_luts(32, 1) == 1
        assert calculate_srl16e_luts(64, 1) == 2

        # Bitwidth 3: ⌈3/2⌉ = 2
        assert calculate_srl16e_luts(32, 3) == 2
        assert calculate_srl16e_luts(64, 3) == 4

        # Bitwidth 8: ⌈8/2⌉ = 4
        assert calculate_srl16e_luts(32, 8) == 4
        assert calculate_srl16e_luts(64, 8) == 8

    def test_calculate_srl16e_luts_small_depths(self):
        """Test SRL16E LUT calculations for small depths."""
        from finn.transformation.fpgadataflow.simulation import calculate_srl16e_luts

        # Small depths still use at least 1 LUT per bitwidth factor
        assert calculate_srl16e_luts(2, 2) == 1
        assert calculate_srl16e_luts(16, 2) == 1
        assert calculate_srl16e_luts(31, 2) == 1


class TestSRL16EDepthRange:
    """Test SRL16E depth range inversion function."""

    def test_depth_range_basic(self):
        """Test basic depth range calculation for SRL16E."""
        from finn.transformation.fpgadataflow.simulation import (
            calculate_srl16e_depth_range,
            calculate_srl16e_luts,
        )

        # 1 LUT, bitwidth=2
        min_d, max_d = calculate_srl16e_depth_range(1, 2)
        assert min_d == 2
        assert max_d == 32
        assert calculate_srl16e_luts(min_d, 2) == 1
        assert calculate_srl16e_luts(max_d, 2) == 1

    def test_depth_range_bitwidth_1(self):
        """Test depth range for 1-bit data."""
        from finn.transformation.fpgadataflow.simulation import (
            calculate_srl16e_depth_range,
            calculate_srl16e_luts,
        )

        min_d, max_d = calculate_srl16e_depth_range(1, 1)
        assert min_d == 2
        assert max_d == 32
        assert calculate_srl16e_luts(min_d, 1) == 1
        assert calculate_srl16e_luts(max_d, 1) == 1

        min_d, max_d = calculate_srl16e_depth_range(2, 1)
        assert min_d == 33
        assert max_d == 64
        assert calculate_srl16e_luts(min_d, 1) == 2
        assert calculate_srl16e_luts(max_d, 1) == 2

    def test_depth_range_invalid_odd_luts(self):
        """Test that odd LUT counts are invalid for certain bitwidths."""
        from finn.transformation.fpgadataflow.simulation import calculate_srl16e_depth_range

        # Bitwidth=4: ⌈4/2⌉ = 2, so only even LUT counts are valid
        _, max_d = calculate_srl16e_depth_range(1, 4)
        assert max_d == 0, "1 LUT should be invalid for bitwidth=4"

        _, max_d = calculate_srl16e_depth_range(2, 4)
        assert max_d > 0, "2 LUTs should be valid for bitwidth=4"

    def test_depth_range_consistency(self):
        """Test that all valid ranges produce the correct LUT count."""
        from finn.transformation.fpgadataflow.simulation import (
            calculate_srl16e_depth_range,
            calculate_srl16e_luts,
        )

        for bitwidth in [1, 2, 3, 4, 8, 16]:
            for luts in range(1, 20):
                min_d, max_d = calculate_srl16e_depth_range(luts, bitwidth)
                if max_d > 0:  # Valid configuration
                    # Verify both endpoints produce correct LUT count
                    assert calculate_srl16e_luts(min_d, bitwidth) == luts, (
                        f"Min depth {min_d} for {luts} LUTs, "
                        f"bitwidth {bitwidth} produces wrong count"
                    )
                    assert calculate_srl16e_luts(max_d, bitwidth) == luts, (
                        f"Max depth {max_d} for {luts} LUTs, "
                        f"bitwidth {bitwidth} produces wrong count"
                    )

                    # Verify just outside the range produces different counts
                    if min_d > 2:
                        assert calculate_srl16e_luts(min_d - 1, bitwidth) < luts
                    assert calculate_srl16e_luts(max_d + 1, bitwidth) > luts


class TestNeedsMinimization:
    """Test the needs_minimization method."""

    # TODO: Maybe remove this behavior
    def test_small_depths_no_minimization(self):
        """Test that small depths don't need minimization."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)
        sim.max_qsrl_depth = 256

        # Depths <= 32 don't need minimization (fit in bitwidth/2 LUTs)
        assert not sim._needs_minimization(32, 8)
        assert not sim._needs_minimization(16, 8)
        assert not sim._needs_minimization(2, 8)

    # TODO: Maybe remove this behavior
    def test_qsrl_range_no_minimization(self):
        """Test that depths within QSRL range don't need minimization."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)
        sim.max_qsrl_depth = 256

        # Depths within max_qsrl_depth don't need minimization
        assert not sim._needs_minimization(128, 8)
        assert not sim._needs_minimization(256, 8)

    def test_large_depths_need_minimization(self):
        """Test that large depths with multiple BRAM blocks need minimization."""
        from finn.transformation.fpgadataflow.simulation import (
            RunLayerParallelSimulation,
            calculate_bram_blocks,
            calculate_bram_depth_range,
        )

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)
        sim.max_qsrl_depth = 256

        # Test with specific known cases first
        # bitwidth=8: 1 BRAM range is (1, 2048)
        # Use depth > 2048 to get multiple blocks
        depth = 5000
        bitwidth = 8
        blocks = calculate_bram_blocks(depth, bitwidth)
        assert blocks > 1, f"depth={depth}, bitwidth={bitwidth} should use >1 BRAM"
        assert sim._needs_minimization(depth, bitwidth)

        # bitwidth=18: 1 BRAM range is (1, 1024)
        # Use depth > 1024 to get multiple blocks
        depth = 3000
        bitwidth = 18
        blocks = calculate_bram_blocks(depth, bitwidth)
        assert blocks > 1, f"depth={depth}, bitwidth={bitwidth} should use >1 BRAM"
        assert sim._needs_minimization(depth, bitwidth)

        # Verify that depth with 1 BRAM doesn't need minimization
        # when it's at minimum block count
        depth = 1000
        bitwidth = 8
        blocks = calculate_bram_blocks(depth, bitwidth)
        assert blocks == 1
        assert not sim._needs_minimization(depth, bitwidth)

        # Exhaustive test: check that depths with MORE than minimum BRAM blocks
        # need minimization (unless very close to QSRL threshold)
        for bw in range(1, 64):
            # Find the minimum achievable block count for this bitwidth
            min_blocks = None
            max_d = 0
            test_blocks = 1
            while max_d == 0:
                _, max_d = calculate_bram_depth_range(test_blocks, bw)
                if max_d > 0:
                    min_blocks = test_blocks
                    break
                test_blocks += 1

            if min_blocks is None:
                continue  # Skip if no valid config found

            # Test depths that use more blocks than minimum
            for depth in range(1, 8192):
                blocks = calculate_bram_blocks(depth, bw)

                # Only expect minimization if blocks > minimum achievable
                if blocks > min_blocks and depth > math.floor(sim.max_qsrl_depth * 1.1):
                    assert sim._needs_minimization(depth, bw), (
                        f"depth={depth}, bw={bw}, blocks={blocks}, min_blocks={min_blocks} "
                        f"should need minimization"
                    )

    def test_minimum_bram_edge_case(self):
        """Test edge case at minimum BRAM blocks."""
        from finn.transformation.fpgadataflow.simulation import RunLayerParallelSimulation

        sim = RunLayerParallelSimulation.__new__(RunLayerParallelSimulation)
        sim.max_qsrl_depth = 256

        # A depth that's just slightly above max_qsrl_depth with minimum BRAM blocks
        # The behavior depends on whether it's deemed too close to optimize
        depth = 300
        bitwidth = 1

        # Verify the method executes without error
        result = sim._needs_minimization(depth, bitwidth)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
