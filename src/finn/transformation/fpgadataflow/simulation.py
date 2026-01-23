"""Manage FINN simulation variants."""

import json
import math
import time
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation, SimulationType
from finn.transformation.fpgadataflow.simulation_controller import (
    NodeConnectedSimulationController,
    NodeIsolatedSimulationController,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import DisabledLoggingConsole, log

if TYPE_CHECKING:
    from onnx.onnx_ml_pb2 import NodeProto

FIFODepthConfig: TypeAlias = dict[int, dict[str, str | list[int]]]
IsoSimData = NodeIsolatedSimulationController.IsolatedSimReturnType


class Simulation:
    """Manage simulation (runs) in FINN. Upon instance creation, the simulation will be built.

    IMPORTANT: If the modelwrapper was somehow changed, create a NEW simulation object!
    """

    def __init__(
        self,
        model: ModelWrapper,
        simulation_type: SimulationType,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        workers: int | None = None,
    ) -> None:
        """Create a new simulation instance. Read simulation binary paths
        from the simulation_binaries metadata prop field."""
        self.simulation_type = simulation_type
        self.model = model
        sim_binaries = self.model.get_metadata_prop("simulation_binaries")

        if sim_binaries is None:
            raise FINNUserError(
                "No field simulation_binaries found in the model. Make "
                "sure to run the BuildSimulation transformation beforehand."
            )
        sim_binaries: list[Path] = [Path(p) for p in str(sim_binaries).split("\n")]
        if len(sim_binaries) != len(self.model.graph.node):
            raise FINNUserError(
                "The number of found simulation binaries does not match the number "
                "of nodes in the graph. Make sure to run BuildSimulation just "
                "before."
            )
        if any(not p.exists() for p in sim_binaries):
            raise FINNUserError(
                "Simulation binary data points to invalid paths. Please rerun BuildSimulation."
            )
        # TODO: Currently we have to recompile even if we just
        # TODO: called BuildSimulation in the step before
        # (However this only compiles, it should NOT stitch the IPs again)
        self.model = self.model.transform(
            BuildSimulation(fpgapart, clk_ns, functional_sim, simulation_type, workers)
        )
        self.binaries: dict[int, Path] = {i: sim_binaries[i] for i in range(len(sim_binaries))}

    def simulate(self, *args: Any, **kwargs: Any) -> Any:
        """Run the built simulation and return its results. This function can always be called
        and will lookup the correct function to use, but consequently
        cannot provide typing information."""
        match self.simulation_type:
            case SimulationType.NODE_BASED_CONNECTED:
                print("Connected simulation")
                return self.simulate_node_connected(*args, **kwargs)
            case SimulationType.NODE_BASED_ISOLATED:
                return self.simulate_node_isolated(*args, **kwargs)
            case _:
                raise FINNUserError(f"Unsupported simulation type {self.simulation_type}")

    def simulate_node_connected(
        self, depth: int | list[list[int]] | None = None, max_cycles: int | None = None
    ) -> tuple[dict[int, dict[str, list[int]]], bool]:
        """Simulate the given number of samples for every layer. Layers are completely isolated
        and simulated in parallel. Simulation data is returned as a dict (by node name as index).
        """
        if self.simulation_type != SimulationType.NODE_BASED_CONNECTED:
            raise FINNInternalError(
                f"Called simulation function 'simulate_node_connected' "
                f"does not match provided simulation type "
                f"{self.simulation_type}"
            )
        names = [node.name for node in self.model.graph.node]
        initial_depth: Any = [[depth]] * len(self.binaries) if isinstance(depth, int) else depth

        # Run simulation
        start = time.time()
        output_json = Path(make_build_dir("simulation_results_")) / "simulation_data.json"
        with DisabledLoggingConsole() as console:
            controller = NodeConnectedSimulationController(
                len(self.binaries), names, list(self.binaries.values()), console, 0.1, False
            )
            controller.run(initial_depth, output_json, max_cycles)
        end = time.time()
        log.info(f"Simulation took {end - start} seconds!")

        # Load the merged data from JSON
        merged_data = json.loads(output_json.read_text())

        # Return the collected data indexed by node index
        data = {}
        for i, sim_entry in enumerate(merged_data["simulations"]):
            data[i] = {
                "name": sim_entry["name"],
                "fifo_utilization": sim_entry["fifo_utilization"],
                "fifo_depth": sim_entry["fifo_depth"],
                "cycles": sim_entry["cycles"],
                "samples": sim_entry["samples"],
                "intervals": sim_entry["intervals"],
            }
        json.dump(data, output_json.open("w"), indent=4)
        return data, merged_data.get("timeout_occurred", False)

    def simulate_node_isolated(self) -> dict[str, IsoSimData]:
        """Simulate isolated nodes."""
        if self.simulation_type != SimulationType.NODE_BASED_ISOLATED:
            raise FINNInternalError(
                f"Called simulation function 'simulate_node_isolated' "
                f"does not match provided simulation type "
                f"{self.simulation_type}"
            )
        names = [node.name for node in self.model.graph.node]
        with DisabledLoggingConsole() as console:
            controller = NodeIsolatedSimulationController(
                len(self.binaries), names, list(self.binaries.values()), console, 0.1, False
            )
            return controller.run()


class RunLayerParallelSimulation(Transformation):  # noqa
    def __init__(
        self,
        fpgapart: str,
        clk_ns: float,
        cfg: DataflowBuildConfig,
        max_qsrl_depth: int = 256,
        vivado_ram_style: str = "auto",
        quality_of_results: str = "default",
    ) -> None:
        """Run layer parallel simulations."""
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth
        self.vivado_ram_style = vivado_ram_style
        self.quality_of_results = quality_of_results

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run layer parallel simulations."""
        sim = Simulation(
            model,
            SimulationType.NODE_BASED_CONNECTED,
            self.fpgapart,
            self.clk_ns,
            self.cfg.functional_simulation,
        )
        model = sim.model  # TODO:clean up

        initial_fifo_depths, _ = sim.simulate()

        fifo_depths = []  # Each entry is a list of fifo sizes for that node
        for val in initial_fifo_depths.values():
            fifo_depths.append([v + 1 for v in val["fifo_utilization"]])

        # Max cycles for any simulation
        sim_cycles = max([val["cycles"] for val in initial_fifo_depths.values()])

        bit_widths = []
        for i in range(len(fifo_depths)):
            bit_widths.append([])
            hw_node = getCustomOp(model.graph.node[i])
            if isinstance(hw_node, HWCustomOp):
                for j in range(len(fifo_depths[i])):
                    bit_widths[i].append(hw_node.get_outstream_width(j))
            else:
                raise FINNInternalError("Non-HW node found in dataflow graph during simulation")

        needs_minimization = []
        for i in range(len(fifo_depths)):
            needs_minimization.append([True] * len(fifo_depths[i]))
        for i in range(len(fifo_depths)):
            for j in range(len(fifo_depths[i])):
                # Check if we can reduce the fifo size

                used_size = fifo_depths[i][j]
                bw = bit_widths[i][j]

                needs_minimization[i][j] = self._needs_minimization(used_size, bw)

        # Preserve original baseline depths for testing (deep copy)
        original_fifo_depths = [row[:] for row in fifo_depths]

        # Minimize FIFO depths using binary search over BRAM block counts
        for i in range(len(fifo_depths)):
            for j in range(len(fifo_depths[i])):
                if not needs_minimization[i][j]:
                    continue

                minimized_depth = self._minimize_fifo_depth(
                    i,
                    j,
                    fifo_depths,
                    original_fifo_depths,
                    bit_widths,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                )
                fifo_depths[i][j] = minimized_depth

        print("Final FIFO depths:")
        for i in range(len(fifo_depths)):
            print(f"{i}: {fifo_depths[i]}")
            log.info(f"{i}: {fifo_depths[i]}")

        # Write back results. By default write to output_dir / "fifo_config.json"
        assert len(fifo_depths) == len(model.graph.node)
        json_results = {}
        for i in range(len(fifo_depths)):
            json_results[i] = {"node": model.graph.node[i].name, "depths": fifo_depths[i]}
        with (Path(self.cfg.output_dir) / "fifo_config.json").open("w") as f:
            json.dump(json_results, f)

        return model, False

    def _check_performance(self, new_data: dict, initial_fifo_depths: dict) -> bool:
        """Check if performance has degraded compared to baseline.

        Args:
            new_data: Simulation results to check
            initial_fifo_depths: Baseline performance data

        Returns:
            True if performance degraded, False otherwise
        """
        for k, v in new_data.items():
            for idx in range(len(v["intervals"])):
                if v["intervals"][idx] > initial_fifo_depths[k]["intervals"][idx]:
                    return True
        return False

    def _test_depth(
        self,
        test_depth: int,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
    ) -> tuple[bool, bool]:
        """Test a specific FIFO depth.

        Args:
            test_depth: Depth to test
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles

        Returns:
            Tuple of (success, timeout) where success means depth works without degradation
        """
        test_depths = [row[:] for row in baseline_depths]  # Deep copy from baseline
        test_depths[node_idx][fifo_idx] = test_depth

        new_data, timeout = sim.simulate_node_connected(
            test_depths, max_cycles=math.ceil(sim_cycles * 1.1)
        )

        if timeout:
            return False, True

        performance_degraded = self._check_performance(new_data, initial_fifo_depths)
        return not performance_degraded, False

    def _get_valid_block_counts(self, min_blocks: int, max_blocks: int, bitwidth: int) -> list[int]:
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

    def _minimize_fifo_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        current_depths: list,
        baseline_depths: list,
        bit_widths: list,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: int,
    ) -> int:
        """Minimize a single FIFO depth using binary search.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration
            (may have already-minimized values)
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bit_widths: Bitwidths for all FIFOs
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles

        Returns:
            Minimized FIFO depth
        """
        original_size = baseline_depths[node_idx][fifo_idx]
        bw = bit_widths[node_idx][fifo_idx]

        print(f"Minimizing Node {node_idx} FIFO {fifo_idx}: original depth {original_size}")

        # If FIFO depth of 32 works, use it because it fits into bw/2 LUTs
        success, timeout = self._test_depth(
            32, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
        )
        if success:
            return 32

        if original_size <= self.max_qsrl_depth:
            upper_luts = calculate_srl16e_luts(original_size, bw)
            # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
            lower_luts = calculate_srl16e_luts(64, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    baseline_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                return best_working_depth
            return original_size

        # Try FIFO depth of 256 next (fits into LUTRAM)
        success, timeout = self._test_depth(
            self.max_qsrl_depth,
            node_idx,
            fifo_idx,
            baseline_depths,
            initial_fifo_depths,
            sim,
            sim_cycles,
        )
        if success:
            upper_luts = calculate_srl16e_luts(original_size, bw)
            # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
            lower_luts = calculate_srl16e_luts(64, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    baseline_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                return best_working_depth
            return self.max_qsrl_depth

        # We know 256 doesn't work, so we have to use BRAMs
        # Try one BRAM block less than current
        upper_blocks = calculate_bram_blocks(original_size, bw)
        # Get all valid block counts in the range
        valid_blocks = self._get_valid_block_counts(1, upper_blocks - 1, bw)
        if not valid_blocks:
            # No valid configurations exist
            return original_size
        # Test the maximum valid block count first (smallest depth)
        max_valid_blocks = valid_blocks[-1]
        _, max_d = calculate_bram_depth_range(max_valid_blocks, bw)

        success, timeout = self._test_depth(
            max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
        )

        if timeout or not success:
            return original_size

        best_working_depth = max_d

        # Binary search if there's room to search and multiple valid configs
        if len(valid_blocks) > 1:
            best_working_depth = self._exponential_binary_search_depth(
                node_idx,
                fifo_idx,
                baseline_depths,
                bw,
                initial_fifo_depths,
                sim,
                sim_cycles,
                valid_blocks=valid_blocks,
            )

        return best_working_depth

    def _exponential_binary_search_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        bitwidth: int,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
        valid_blocks: list[int],
    ) -> int:
        """Perform exponential + binary search over valid block configurations.

        Uses exponential search to quickly find the range, then binary search within it.
        This is more efficient when smaller block counts are more likely.
        Only searches over pre-validated block counts.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            valid_blocks: Sorted list of valid block counts to search over

        Returns:
            Best working depth found
        """
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

            success, _ = self._test_depth(
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

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

            success, _ = self._test_depth(
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

            if success:
                # This depth works, try smaller (lower indices)
                best_working_depth = max_d
                upper_idx = mid_idx
            else:
                # This depth doesn't work, need larger (higher indices)
                lower_idx = mid_idx + 1

        return best_working_depth

    def _binary_search_srl_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        bitwidth: int,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
        lower_luts: int,
        upper_luts: int,
    ) -> int:
        """Perform binary search to find minimal working FIFO depth in LUTRAM range.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            lower_luts: Lower bound for LUT count
            upper_luts: Upper bound for LUT count (known to work)

        Returns:
            Best working depth found
        """
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

            success, _ = self._test_depth(
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

            if success:
                # This depth works, try smaller
                best_working_depth = max_d
                upper_luts = mid_luts
            else:
                # This depth doesn't work, need larger
                lower_luts = mid_luts + 1

        return best_working_depth

    def _needs_minimization(self, fifo_depth: int, bitwidth: int) -> bool:
        """Determine whether a FIFO can be minimized further.

        Args:
            fifo_depth: Current FIFO depth
            bitwidth: Data bitwidth

        Returns:
            True if the FIFO can be minimized further, False otherwise.
        """
        # Qsrl FIFO Formula: LUTs = ⌈depth/32⌉ x ⌈bitwidth/2⌉
        if fifo_depth <= 32:  # FIFOs of depth <=32 fit into bitwidth/2 LUTs
            return False
        # Return False if exactly the minimum number of possible BRAM blocks is used for this
        # bitwidth and depth is sufficiently large that further optimization is unlikely to succeed
        return not (
            calculate_bram_blocks(fifo_depth, bitwidth)
            <= self._get_valid_block_counts(1, bitwidth, bitwidth)[0]
            and fifo_depth > math.floor(self.max_qsrl_depth * 1.1)
        )


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


class RunLayerIsolatedSimulation(Transformation):
    """Run a layer isolated simulation and calculate some information for a
    later layer parallel simulation."""

    def __init__(self, fpgapart: str, clk_ns: float, functional_sim: bool) -> None:
        """Run isolated layer simulations."""
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.functional_sim = functional_sim

    def calculate_upper_bounds(self, data: IsoSimData) -> dict[str, int]:
        """Try to calculate an upper bound for the incoming FIFO size of the layers.
        Return size indexed by node name."""
        # First get the input ready signals of all layers
        # TODO: We assume that every cycle gets recorded here
        readies: dict[str, list[int]] = {
            name: data[name]["ready"].values()[2] for name in data.keys()
        }

        # Calculate the count of _not_ ready cycles between the
        # first ready and the first ready of the second sample
        # TODO: Currently we simply divide target cycles by 2, since
        # TODO: this is multiplied on the C++ side, but this may change in the
        # TODO: future
        cycles_per_sample: dict[str, int] = {}
        for name in data.keys():
            cycles: int = data[name]["ready"].values()[1]
            if cycles % 2 != 0:
                raise FINNInternalError(
                    f"Layer {name} has an odd number "
                    f"of ready cycles per sample. This points "
                    f"towards a change in the C++ version, "
                    f"since we currently assume that the number "
                    f"we get here is twice the number per sample "
                    f"(since we want to simulate 2 samples). "
                    f"Getting this error might indicate that this "
                    f"has to be fixed."
                )
            cycles_per_sample[name] = cycles / 2

        # TODO: This calculation assumes, that if the producer does NOT fire the entire time,
        # TODO: the consumer can read at least at the same speed as
        #       if the producer did, and not slower.
        # TODO: (Since this would mean that less data pressure from
        #       the producer makes the consumer _slower_.)
        # TODO: This should usually be the case, but is important to keep in mind.
        return {
            # num of zeroes = num total - num of ones
            name: len(readies[name])
            - sum(readies[name])
        }

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run isolated layer simulations."""
        sim = Simulation(
            model,
            SimulationType.NODE_BASED_ISOLATED,
            self.fpgapart,
            self.clk_ns,
            self.functional_sim,
        )
        data: dict[str, IsoSimData] = sim.simulate_node_isolated()
        in_fifo_upper_bound = self.calculate_upper_bounds(data)
        formatted_upper_bounds = "\n\t".join(
            [f"{name}: {in_fifo_upper_bound[name]}" for name in in_fifo_upper_bound.keys()]
        )
        log.info("Upper bounds: \n" + formatted_upper_bounds)

        raise NotImplementedError()
        # TODO: Integrate data into the layer parallel simulation
        return model, False


class ApplyFIFOSizes(Transformation):
    """Apply a FIFO sizing configuration to the model. If not existing, inserts FIFOs beforehand."""

    def __init__(
        self,
        cfg: DataflowBuildConfig,
        fifo_config: Path | None = None,
        max_qsrl_depth: int = 256,
        vivado_ram_style: str = "auto",
    ) -> None:
        """If given read the config json from the given path.
        Otherwise check in the output directory.
        """
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth
        self.vivado_ram_style = vivado_ram_style
        if fifo_config is None:
            self.path = Path(cfg.output_dir) / "fifo_config.json"
        else:
            self.path = fifo_config

        self.depth: FIFODepthConfig = {}
        with self.path.open() as f:
            self.depth = cast("FIFODepthConfig", json.load(f))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply FIFO Simulation Depths to the model."""
        # TODO: Better way to check for fifos (op_type for example)
        if len(list(filter(lambda node: "StreamingFIFO" in node.op_type, model.graph.node))) > 0:
            log.warning(
                "It seems that StreamingFIFOs have already "
                "been inserted into the graph. Skipping insertion of FIFOs."
            )
        else:
            if len(model.graph.node) != len(self.depth):
                raise FINNUserError(
                    "There are no StreamingFIFOs in the graph, yet the number "
                    "of nodes and number of FIFO settings differ. There may be "
                    "unaccounted for nodes that have not been part of the FIFO "
                    "simulation. Consider re-running simulation directly before "
                    "applying the FIFO sizes. It might also be that your model "
                    "or config is outdated, in which case it is recommended to "
                    "re-run the entire flow from start to finish."
                )

            # Inser the FIFOs into the model
            model = model.transform(InsertFIFO(True, self.max_qsrl_depth, self.vivado_ram_style))

            # Synthesize the nodes (TODO: Remove)
            model = model.transform(GiveUniqueNodeNames())
            model: ModelWrapper = model.transform(GiveReadableTensorNames())
            model = model.transform(SpecializeLayers(self.cfg._resolve_fpga_part()))  # noqa
            model = model.transform(GiveUniqueNodeNames())
            model: ModelWrapper = model.transform(GiveReadableTensorNames())
            model = model.transform(
                PrepareIP(
                    fpgapart=self.cfg._resolve_fpga_part(),
                    clk=self.cfg.synth_clk_period_ns,  # noqa
                )
            )
            model = model.transform(HLSSynthIP())

            # Sanity check to make sure fifos were inserted
            inserted_fifo_count = sum(
                [int("StreamingFIFO" in node.op_type) for node in model.graph.node]
            )
            if inserted_fifo_count == 0:
                raise FINNInternalError(
                    "No FIFOs were inserted. This may be due to "
                    "wrong network configuration, step order or "
                    "a number of other things."
                )
            if inserted_fifo_count < int(0.1 * float(len(model.graph.node))):
                log.warning(
                    "The number of inserted FIFOs makes up less than 10%"
                    " of the total number of nodes in the model. This could "
                    "point to a potential error."
                )

            # Assign data based on the names of the nodes. Since no FIFOs were in the graph
            # before, the names should stay the same.
            # TODO: This currently assumes that the FIFO to be sized comes AFTER the actual node.
            # TODO: If the simulation code is changed, this needs to be changed as well
            for i in range(len(model.graph.node)):
                node: NodeProto = model.graph.node[i]
                node_inst: HWCustomOp = getCustomOp(model.graph.node[i])
                if node.op_type.startswith("StreamingFIFO"):
                    # FIFOs can only have one producer, so this must
                    # be the node whoose simulated depth we have to get
                    predecessors: list[NodeProto] | None = model.find_direct_predecessors(node)
                    if predecessors is not None and len(predecessors) > 1:
                        raise FINNInternalError(f"FIFO node {node.name} has multiple producers!")
                    if predecessors is None:
                        continue
                    predecessor: NodeProto = predecessors[0]

                    # Check which of the predecessors outputs this FIFO is connected to
                    # and use the depth at that index from the simulation
                    for sim_node_name, sim_depths in self.depth.values():
                        if sim_node_name == predecessor.name:
                            depth_index = predecessor.output.index(node.input[0])
                            depth = int(sim_depths[depth_index])
                            node_inst.set_nodeattr("depth", depth)

                            # TODO: Code copied from old FIFO sizing
                            # exception for top-level IO FIFOs which cause a bug in simulation
                            # (top-level IOs should not have impl_style=vivado)
                            toplevel_in = node.input[0] in [x.name for x in model.graph.input]
                            toplevel_out = node.output[0] in [x.name for x in model.graph.output]
                            toplevel_style_exception = toplevel_in or toplevel_out
                            # Set FIFO implementation/ram styles
                            if (depth > self.max_qsrl_depth) and (not toplevel_style_exception):
                                node_inst.set_nodeattr("impl_style", "vivado")
                                node_inst.set_nodeattr("ram_style", self.vivado_ram_style)
                            else:
                                node_inst.set_nodeattr("impl_style", "rtl")

            # TODO: Following code is copied from the old FIFO sizing. Might be shortenable
            for node in model.graph.node:
                if not node.op_type.startswith("StreamingFIFO"):
                    node_inst = getCustomOp(node)
                    fifodepth_in = []
                    for node_inp in node.input:
                        prod = model.find_producer(node_inp)
                        if prod is None:
                            # no producer for this input
                            if node_inp in [x.name for x in model.graph.input]:
                                # top-level input with no FIFO
                                fifodepth_in.append(0)
                            else:
                                # FIFO depth attr applies only to dynamic attributes
                                pass
                        else:
                            # there is a producer for this input
                            if prod.op_type.startswith("StreamingFIFO"):
                                prod_inst = getCustomOp(prod)
                                fifodepth_in.append(prod_inst.get_nodeattr("depth"))
                            else:
                                # explicitly no FIFO on this dynamic input
                                fifodepth_in.append(0)
                    fifodepth_out = []
                    for node_out in node.output:
                        cons = model.find_consumer(node_out)
                        if cons is None:
                            # no consumer for this output
                            if node_out in [x.name for x in model.graph.output]:
                                # top-level output with no FIFO
                                fifodepth_out.append(0)
                            else:
                                # FIFO depth attr applies only to dynamic attributes
                                pass
                        else:
                            # there is a consumer for this input
                            if cons.op_type.startswith("StreamingFIFO"):
                                cons_inst = getCustomOp(cons)
                                fifodepth_out.append(cons_inst.get_nodeattr("depth"))
                            else:
                                # explicitly no FIFO on this dynamic output
                                fifodepth_out.append(0)
                    node_inst.set_nodeattr("inFIFODepths", fifodepth_in)
                    node_inst.set_nodeattr("outFIFODepths", fifodepth_out)

        # Synthesize with the proper sizes set
        model = model.transform(SpecializeLayers(self.cfg._resolve_fpga_part()))  # noqa
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(
            PrepareIP(
                fpgapart=self.cfg._resolve_fpga_part(),
                clk=self.cfg.synth_clk_period_ns,  # noqa
            )
        )
        model = model.transform(HLSSynthIP())
        # model.set_metadata_prop("rtlsim_trace", "")
        # model.set_metadata_prop("rtlsim_so", "")
        # model.set_metadata_prop("vivado_stitch_proj", "")
        # model.set_metadata_prop("wrapper_filename", "")
        # model.set_metadata_prop("vivado_stitch_vlnv", "")
        # model.set_metadata_prop("vivado_stitch_ifnames", "")
        # model.set_metadata_prop("exec_mode", "")

        return model, False
