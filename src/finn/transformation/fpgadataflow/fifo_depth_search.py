# Copyright (C) 2025-2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Simulator-independent parts of the per-FIFO depth minimisation.

The distributed RTL simulation (``simulation_connected``) and the abstract TEG simulation
(``finn.analysis.fpgadataflow.teg.search``) run the same block-granular search per FIFO: test
depth 32, binary search over SRL16E LUT counts up to ``max_qsrl_depth``, then exponential plus
binary search over valid BRAM block counts. The two strategies differ only in the oracle that
decides whether a candidate depth sustains the target throughput, which is passed in as a
callable ``test_depth(depth) -> (success, timeout)``.

This module holds the cost model (SRL16E LUTs, BRAM/URAM blocks), the candidate generation
and the search skeleton so that both strategies produce comparable depths.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum

from finn.transformation.fpgadataflow.set_fifo_depths import get_fifo_split_configs
from finn.util.exception import FINNInternalError

#: Hardware BRAM FIFOs lose entries to internal pipeline registers compared to the software FIFO
#: model (which has exact capacity). This constant accounts for that overhead so that the
#: minimization algorithm finds depths that are safe to deploy on hardware.
BRAM_FIFO_PIPELINE_OVERHEAD = 2

#: The result of ``test_depth``: (success, timeout)
DepthTest = Callable[[int], tuple[bool, bool]]


class MinimizationOrder(Enum):
    """The order in which the search algorithm minimizes the FIFO depths."""

    NODE_ORDER = 0
    REVERSE_NODE_ORDER = 1
    LARGEST_BITWIDTH_DIFF_FIRST = 2
    SMALLEST_BITWIDTH_DIFF_FIRST = 3

    # Non black-box model orders
    AFTER_THRESHOLDS_FIRST = 4
    AFTER_DWC_FIRST = 5

    # Half black-box
    # If we ran a sim before, we know the largest FIFOs, so start with these.
    # This strategy might work, if the changes to the model are small enough
    REUSE_PREVIOUS_ORDER = 6


# --------------------------------------------------------------------------- BRAM overhead
def count_bram_sub_fifos(depth: int, max_qsrl_depth: int) -> int:
    """Return the number of BRAM (vivado) sub-FIFOs that *depth* decomposes into.

    Non-power-of-two BRAM FIFOs are decomposed into several power-of-two sub-FIFOs by
    get_fifo_split_configs.  Each sub-FIFO whose style is "vivado" has its own pipeline
    register overhead, so the total overhead scales with the sub-FIFO count.
    """
    return sum(1 for _, style in get_fifo_split_configs(depth, max_qsrl_depth) if style == "vivado")


def effective_capacity(depth: int, max_qsrl_depth: int) -> int:
    """Token capacity the hardware FIFO of nominal ``depth`` actually provides."""
    if depth <= max_qsrl_depth:
        return depth
    return depth - count_bram_sub_fifos(depth, max_qsrl_depth) * BRAM_FIFO_PIPELINE_OVERHEAD


def safe_bram_starting_depth(peak_util: int, max_qsrl_depth: int) -> int:
    """Return the smallest depth d such that d minus its BRAM pipeline overhead >= peak_util + 1.

    For LUTRAM depths (d <= max_qsrl_depth) the software model is exact so no overhead is needed.
    For BRAM depths the overhead depends on how many sub-FIFOs the decomposition produces,
    which itself depends on d.  We iterate (typically 1-2 steps) until the overhead stabilises.
    """
    d = max(peak_util + 1, 32)
    if d <= max_qsrl_depth:
        return d
    # Iteratively find d where d - num_vivado(d)*overhead >= peak_util + 1
    overhead = 0
    while True:
        d = peak_util + 1 + overhead
        num_vivado = count_bram_sub_fifos(d, max_qsrl_depth)
        new_overhead = num_vivado * BRAM_FIFO_PIPELINE_OVERHEAD
        if new_overhead <= overhead:
            break
        overhead = new_overhead
    return max(d, 32)


# --------------------------------------------------------------------------- cost model
def calculate_bram_blocks(depth: int, bitwidth: int) -> int:
    """Calculate the number of BRAM blocks required for a BRAM FIFO.

    Args:
        depth: FIFO depth
        bitwidth: Data bitwidth
    """
    if bitwidth == 1:
        return math.ceil(depth / 16384)
    if bitwidth == 2:
        return math.ceil(depth / 8192)
    if bitwidth <= 4:
        return (math.ceil(depth / 4096)) * (math.ceil(bitwidth / 4))
    if bitwidth <= 9:
        return (math.ceil(depth / 2048)) * (math.ceil(bitwidth / 9))
    if bitwidth <= 18 or depth > 512:
        return (math.ceil(depth / 1024)) * (math.ceil(bitwidth / 18))
    return (math.ceil(depth / 512)) * (math.ceil(bitwidth / 36))


def calculate_bram_depth_range(blocks: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of BRAM blocks.

    Args:
        blocks: Number of BRAM blocks
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'blocks' BRAM blocks.
    """
    if blocks < 1:
        raise FINNInternalError("Number of BRAM blocks must be at least 1")

    # Invert the formula from calculate_bram_blocks based on bitwidth
    if bitwidth == 1:
        # blocks = ⌈depth/16384⌉
        # Inversion: (blocks-1)*16384 < depth ≤ blocks*16384
        min_depth = (blocks - 1) * 16384 + 1 if blocks > 1 else 1
        max_depth = blocks * 16384
    elif bitwidth == 2:
        # blocks = ⌈depth/8192⌉
        # Inversion: (blocks-1)*8192 < depth ≤ blocks*8192
        min_depth = (blocks - 1) * 8192 + 1 if blocks > 1 else 1
        max_depth = blocks * 8192
    elif bitwidth <= 4:
        # blocks = ⌈depth/4096⌉ * ⌈bitwidth/4⌉
        bitwidth_factor = math.ceil(bitwidth / 4)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 4096 + 1 if depth_blocks > 1 else 1
        max_depth = depth_blocks * 4096
    elif bitwidth <= 9:
        # blocks = ⌈depth/2048⌉ * ⌈bitwidth/9⌉
        bitwidth_factor = math.ceil(bitwidth / 9)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 2048 + 1 if depth_blocks > 1 else 1
        max_depth = depth_blocks * 2048
    elif bitwidth <= 18:
        # blocks = ⌈depth/1024⌉ * ⌈bitwidth/18⌉
        bitwidth_factor = math.ceil(bitwidth / 18)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 1024 + 1
        max_depth = depth_blocks * 1024
    else:
        # bitwidth > 18, split into two cases from original function
        # Case 1: depth > 512 uses ⌈depth/1024⌉ * ⌈bitwidth/18⌉
        # Case 2: depth ≤ 512 uses ⌈depth/512⌉ * ⌈bitwidth/36⌉

        # Try the depth > 512 case first (⌈depth/1024⌉ * ⌈bitwidth/18⌉)
        bitwidth_factor = math.ceil(bitwidth / 18)
        depth_blocks = math.ceil(blocks / bitwidth_factor)

        # Check if blocks is achievable with this bitwidth factor
        if blocks % bitwidth_factor != 0 or depth_blocks < 1:
            # Try the depth ≤ 512 case instead
            pass
        else:
            min_depth = max((depth_blocks - 1) * 1024 + 1, 513)  # Must be > 512
            max_depth = depth_blocks * 1024
            # Check if this range is valid (entirely > 512)
            if min_depth > 512 and calculate_bram_blocks(min_depth, bitwidth) == blocks:
                return (min_depth, max_depth)

        # Try the depth ≤ 512 case (⌈depth/512⌉ * ⌈bitwidth/36⌉)
        bitwidth_factor = math.ceil(bitwidth / 36)
        depth_blocks = math.ceil(blocks / bitwidth_factor)

        # Check if blocks is achievable with this bitwidth factor
        if blocks % bitwidth_factor != 0 or depth_blocks < 1:
            return (0, 0)  # Invalid block count for this bitwidth

        min_depth = (depth_blocks - 1) * 512 + 1 if depth_blocks > 1 else 1
        max_depth = min(depth_blocks * 512, 512)  # Must be ≤ 512

        # Verify the range is valid (entirely ≤ 512 and produces correct block count)
        if max_depth <= 512 and calculate_bram_blocks(min_depth, bitwidth) == blocks:
            return (min_depth, max_depth)

        return (0, 0)  # No valid range found

    # Verify the range is valid
    if calculate_bram_blocks(min_depth, bitwidth) != blocks:
        raise FINNInternalError("Calculated BRAM depth range is invalid!")
    return (min_depth, max_depth)


def calculate_uram_blocks(depth: int, bitwidth: int) -> int:
    """Calculate the number of URAM blocks required for a URAM FIFO.

    Args:
        depth: FIFO depth
        bitwidth: Data bitwidth
    """
    return (math.ceil(depth / 4096)) * (math.ceil(bitwidth / 72))


def calculate_uram_depth_range(blocks: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of URAM blocks.

    Args:
        blocks: Number of URAM blocks
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'blocks' URAM blocks.
        Returns (0, 0) if no valid range exists.
    """
    if blocks < 1:
        return (0, 0)

    # URAM formula: blocks = ⌈depth/4096⌉ * ⌈bitwidth/72⌉
    bitwidth_factor = math.ceil(bitwidth / 72)

    # Calculate depth range
    # Minimum depth: (blocks / bitwidth_factor - 1) * 4096 + 1
    # Maximum depth: (blocks / bitwidth_factor) * 4096

    if blocks % bitwidth_factor != 0:
        return (0, 0)  # Invalid block count for this bitwidth

    depth_blocks = blocks // bitwidth_factor
    min_depth = (depth_blocks - 1) * 4096 + 1 if depth_blocks > 1 else 1
    max_depth = depth_blocks * 4096

    # Verify
    if calculate_uram_blocks(min_depth, bitwidth) != blocks:
        return (0, 0)

    return (min_depth, max_depth)


def calculate_srl16e_luts(depth: int, bitwidth: int) -> int:
    """Calculate the number of SRL16E LUTs required for a FIFO.

    Args:
        depth: FIFO depth (must be >= 2)
        bitwidth: Data bitwidth

    Returns:
        Number of SRL16E LUTs required without adress LUTs.

    Formula: LUTs = ⌈depth/32⌉ x ⌈bitwidth/2⌉
    """
    ram_luts = (math.ceil(depth / 32)) * (math.ceil(bitwidth / 2))
    return ram_luts


def calculate_srl16e_depth_range(luts: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of SRL16E LUTs.

    Args:
        luts: Number of SRL16E LUTs
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'luts' LUTs.
        Returns (0, 0) if no valid range exists.
    """
    if luts < 1:
        return (0, 0)

    # SRL16E formula: luts = ⌈depth/32⌉ * ⌈bitwidth/2⌉
    bitwidth_factor = math.ceil(bitwidth / 2)

    # Calculate depth range
    if luts % bitwidth_factor != 0:
        return (0, 0)  # Invalid LUT count for this bitwidth

    depth_blocks = luts // bitwidth_factor
    min_depth = (depth_blocks - 1) * 32 + 1 if depth_blocks > 1 else 2
    max_depth = depth_blocks * 32

    # Verify
    if calculate_srl16e_luts(min_depth, bitwidth) != luts:
        return (0, 0)

    return (min_depth, max_depth)


# --------------------------------------------------------------------------- candidates
def get_valid_block_counts(min_blocks: int, max_blocks: int, bitwidth: int) -> list[int]:
    """Get all valid BRAM block counts in the specified range.

    Some block counts are invalid for certain bitwidths due to quantization.
    This method returns only the valid configurations.

    Args:
        min_blocks: Minimum block count (inclusive)
        max_blocks: Maximum block count (inclusive)
        bitwidth: Data bitwidth

    Returns:
        Sorted list of valid block counts
    """
    valid_blocks = []
    for blocks in range(min_blocks, max_blocks + 1):
        _, max_d = calculate_bram_depth_range(blocks, bitwidth)
        if max_d > 0:  # Valid configuration
            valid_blocks.append(blocks)
    return valid_blocks


def needs_minimization(fifo_depth: int, bitwidth: int, max_qsrl_depth: int) -> bool:
    """Determine whether a FIFO can be minimized further.

    Args:
        fifo_depth: Current FIFO depth
        bitwidth: Data bitwidth
        max_qsrl_depth: Largest depth implemented as SRL (LUTRAM) FIFO

    Returns:
        True if the FIFO can be minimized further, False otherwise.
    """
    # Qsrl FIFO Formula: LUTs = ⌈depth/32⌉ x ⌈bitwidth/2⌉
    if fifo_depth <= 32:  # FIFOs of depth <=32 fit into bitwidth/2 LUTs
        return False
    # Return False if exactly the minimum number of possible BRAM blocks is used for this
    # bitwidth and depth is sufficiently large that further optimization is unlikely to succeed
    min_blocks = get_valid_block_counts(1, bitwidth, bitwidth)[0]
    return not (
        calculate_bram_blocks(fifo_depth, bitwidth) <= min_blocks
        and fifo_depth > math.floor(max_qsrl_depth * 1.1)
    )


def round_up_to_full_bram_block(depth: int, bitwidth: int, max_qsrl_depth: int) -> int:
    """Round a BRAM depth up to the largest depth using the same number of blocks.

    Partial blocks are not supported by Vivado HLS, so the distributed simulation applies this
    to every final depth above ``max_qsrl_depth``.
    """
    if depth <= max_qsrl_depth:
        return depth
    blocks = calculate_bram_blocks(depth, bitwidth)
    _, max_d = calculate_bram_depth_range(blocks, bitwidth)
    return max_d


def candidate_depths(bitwidth: int, upper: int, max_qsrl_depth: int = 256) -> list[int]:
    """All depths the block-granular search can return for a FIFO of this width, up to ``upper``.

    These are 32 and the largest depth of every SRL16E LUT count up to ``max_qsrl_depth``
    (multiples of 32), followed by the largest depth of every valid BRAM block count, plus
    ``upper`` itself (the safe starting depth is always a valid result).
    """
    cands: set[int] = set()
    d = 32
    while d <= min(upper, max_qsrl_depth):
        cands.add(d)
        d += 32
    if upper > max_qsrl_depth:
        cands.add(max_qsrl_depth)
        upper_blocks = calculate_bram_blocks(upper, bitwidth)
        for blocks in get_valid_block_counts(1, upper_blocks, bitwidth):
            _, max_d = calculate_bram_depth_range(blocks, bitwidth)
            if max_qsrl_depth < max_d <= upper:
                cands.add(max_d)
    cands.add(upper)
    return sorted(cands)


# --------------------------------------------------------------------------- search skeleton
def binary_search_srl_depth(
    test_depth: DepthTest, bitwidth: int, lower_luts: int, upper_luts: int
) -> tuple[int, int]:
    """Perform binary search to find minimal working FIFO depth in LUTRAM range.

    Args:
        test_depth: oracle ``depth -> (success, timeout)``
        bitwidth: Data bitwidth
        lower_luts: Lower bound for LUT count
        upper_luts: Upper bound for LUT count (known to work)

    Returns:
        Tuple: Best working depth found, Number of Iterations required to arrive at this result
    """
    iterations = 0
    _, max_d = calculate_srl16e_depth_range(upper_luts, bitwidth)
    best_working_depth = max_d

    while lower_luts < upper_luts:
        mid_luts = (lower_luts + upper_luts) // 2

        # Prevent infinite loop
        if mid_luts == upper_luts:
            mid_luts = upper_luts - 1
        if mid_luts < lower_luts:
            break

        # Find valid depth for this LUT count
        _, max_d = calculate_srl16e_depth_range(mid_luts, bitwidth)

        if max_d == 0:
            # No valid configuration, try more LUTs
            lower_luts = mid_luts + 1
            continue

        success, _ = test_depth(max_d)
        iterations += 1

        if success:
            # This depth works, try smaller
            best_working_depth = max_d
            upper_luts = mid_luts
        else:
            # This depth doesn't work, need larger
            lower_luts = mid_luts + 1

    return best_working_depth, iterations


def exponential_binary_search_depth(
    test_depth: DepthTest, bitwidth: int, valid_blocks: list[int]
) -> tuple[int, int]:
    """Perform exponential + binary search over valid block configurations.

    Uses exponential search to quickly find the range, then binary search within it.
    This is more efficient when smaller block counts are more likely.
    Only searches over pre-validated block counts.

    Args:
        test_depth: oracle ``depth -> (success, timeout)``
        bitwidth: Data bitwidth
        valid_blocks: Sorted list of valid block counts to search over; the largest is known
            to work.

    Returns:
        Tuple: Best working depth found, Number of iterations required to arrive at this result.
    """
    iterations = 0
    if not valid_blocks:
        raise FINNInternalError("valid_blocks list cannot be empty")

    # Start with the largest valid block count (known to work from caller)
    _, max_d = calculate_bram_depth_range(valid_blocks[-1], bitwidth)
    best_working_depth = max_d

    # Exponential search phase: find range where solution exists
    # Check positions: 0, 1, 2, 4, 8, ... indices in valid_blocks list
    lower_idx = 0
    upper_idx = len(valid_blocks) - 1
    exp_idx = 0
    last_failed_idx = -1

    while exp_idx < upper_idx:
        blocks = valid_blocks[exp_idx]
        _, max_d = calculate_bram_depth_range(blocks, bitwidth)

        success, _ = test_depth(max_d)
        iterations += 1

        if success:
            # Found a working depth, now binary search in [last_failed_idx+1, exp_idx]
            best_working_depth = max_d
            lower_idx = last_failed_idx + 1
            upper_idx = exp_idx
            break
        # This doesn't work, try exponentially larger index
        last_failed_idx = exp_idx
        exp_idx = min(exp_idx * 2 if exp_idx > 0 else 1, upper_idx)

    # Binary search phase: refine the range
    while lower_idx < upper_idx:
        mid_idx = (lower_idx + upper_idx) // 2
        blocks = valid_blocks[mid_idx]
        _, max_d = calculate_bram_depth_range(blocks, bitwidth)

        success, _ = test_depth(max_d)
        iterations += 1

        if success:
            # This depth works, try smaller (lower indices)
            best_working_depth = max_d
            upper_idx = mid_idx
        else:
            # This depth doesn't work, need larger (higher indices)
            lower_idx = mid_idx + 1

    return best_working_depth, iterations


def minimize_fifo_depth(
    original_size: int, bitwidth: int, test_depth: DepthTest, max_qsrl_depth: int = 256
) -> tuple[int, int]:
    """Minimize a single FIFO depth with the block-granular search.

    Args:
        original_size: safe starting depth (known to work)
        bitwidth: Data bitwidth
        test_depth: oracle ``depth -> (success, timeout)``; success means the depth sustains
            the target throughput without deadlock
        max_qsrl_depth: Largest depth implemented as SRL (LUTRAM) FIFO

    Returns:
        Tuple: Minimized FIFO depth, Iterations required to arrive at the result
    """
    iterations = 0

    # If FIFO depth of 32 works, use it because it fits into bw/2 LUTs
    success, _timeout = test_depth(32)
    iterations += 1
    if success:
        return 32, iterations

    if original_size <= max_qsrl_depth:
        upper_luts = calculate_srl16e_luts(original_size, bitwidth)
        # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
        lower_luts = calculate_srl16e_luts(64, bitwidth)

        # Binary search if there's room to search
        if upper_luts > lower_luts:
            best_working_depth, bin_it = binary_search_srl_depth(
                test_depth, bitwidth, lower_luts=lower_luts, upper_luts=upper_luts
            )
            iterations += bin_it
            return best_working_depth, iterations
        return original_size, iterations

    # Try FIFO depth of max_qsrl_depth (256) next (fits into LUTRAM)
    success, _timeout = test_depth(max_qsrl_depth)
    iterations += 1
    if success:
        upper_luts = calculate_srl16e_luts(max_qsrl_depth, bitwidth)
        # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
        lower_luts = calculate_srl16e_luts(64, bitwidth)

        # Binary search if there's room to search
        if upper_luts > lower_luts:
            best_working_depth, bin_it = binary_search_srl_depth(
                test_depth, bitwidth, lower_luts=lower_luts, upper_luts=upper_luts
            )
            iterations += bin_it
            return best_working_depth, iterations
        return max_qsrl_depth, iterations

    # We know 256 doesn't work, so we have to use BRAMs
    # Try one BRAM block less than current
    upper_blocks = calculate_bram_blocks(original_size, bitwidth)
    # Get all valid block counts in the range
    valid_blocks = get_valid_block_counts(1, upper_blocks - 1, bitwidth)
    if not valid_blocks:
        # No valid configurations exist
        return original_size, iterations
    # Test the maximum valid block count first
    # (largest depth below original, most likely to succeed)
    max_valid_blocks = valid_blocks[-1]
    _, max_d = calculate_bram_depth_range(max_valid_blocks, bitwidth)

    success, timeout = test_depth(max_d)
    iterations += 1

    if timeout or not success:
        return original_size, iterations

    best_working_depth = max_d

    # Binary search if there's room to search and multiple valid configs
    if len(valid_blocks) > 1:
        best_working_depth, bin_it = exponential_binary_search_depth(
            test_depth, bitwidth, valid_blocks=valid_blocks
        )
        iterations += bin_it

    return best_working_depth, iterations
