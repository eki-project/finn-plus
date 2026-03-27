"""Manages the Simulation superclass as well as general simulation related transforms."""

import json
import pandas as pd
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from typing import Any, TypeAlias, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation, SimulationType
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

FIFODepthConfig: TypeAlias = list[dict[str, list[int]]]


def store_fifo_data(
    model: ModelWrapper,
    data: pd.DataFrame,
    default_path: Path,
    delete_existing: bool,
    sort_on: str = "onnx_index",
    merge_on: list[str] | None = None,
    merge_how: str = "inner",
    store_html: bool = True,
) -> ModelWrapper:
    """Store the given dataframe in a CSV file.

    If the model already points to data, merge with it and store at the
    path used before (unless delete_existing=True, then simply overwrite at that same path).
    If no data is stored beforehand, use the `default_path` and simply store
    the data there. The path is then entered into the `"fifo_data_path"` metadata prop of the model.

    The function can be used to aggregate benchmarking data across several flow steps.

    Args:
        model: The model that we check for a path to existing FIFO data.
        data: The data to store.
        default_path: Path to use in case that the model doesn't reference a data file yet.
           Is then stored as a metadata prop in the model.
        delete_existing: If true, delete the table and start a new one.
        sort_on: The column to sort on after merging.
        merge_on: What columns to merge on. If "None", use `["onnx_index", "node", "stream"]`
        merge_how: How to merge. Forwarded to pd.merge().
        store_html: If True, also store the data as a HTML with the same name next to the CSV.

    Returns:
        model: Return the model since we might have modified its metadata.
    """
    # Check if all layers are accounted for
    # Note: data may have multiple rows per node (one per output stream)
    if "node" in data.columns:
        num_unique_nodes = len(data["node"].unique())
        if num_unique_nodes != len(model.graph.node):
            raise FINNInternalError(
                f"Tried storing FIFO data for {num_unique_nodes} unique nodes "
                f"but expected {len(model.graph.node)}"
            )
    elif len(data.index) != len(model.graph.node):
        raise FINNInternalError(
            f"Tried storing FIFO data for {len(data.index)} "
            f"values but expected {len(model.graph.node)}"
        )
    fifo_data_path = model.get_metadata_prop("fifo_data_path")
    if fifo_data_path is not None:
        if not fifo_data_path.endswith(".csv"):
            raise FINNInternalError(
                f"It seems the model saved path to store "
                f"the dataframe does not point to a csv file: {fifo_data_path}"
            )
        if delete_existing:
            Path(fifo_data_path).unlink(missing_ok=True)
            merged = data
        else:
            merged = pd.merge(
                data, pd.read_csv(fifo_data_path), on=merge_on, how=merge_how  # type: ignore
            )
            merged = merged.sort_values(sort_on)
        merged.to_csv(fifo_data_path, index=False)
        if store_html:
            merged.to_html(fifo_data_path.replace(".csv", ".html"))
        log.info(f"Stored FIFO dataframe to {fifo_data_path}.")
    else:
        if not default_path.suffix == ".csv":
            raise FINNInternalError(
                f"It seems the provided default path to store "
                f"the dataframe does not point to a csv file: {fifo_data_path}"
            )
        if delete_existing:
            default_path.unlink(missing_ok=True)
        data.to_csv(default_path, index=False)
        if store_html:
            data.to_html(str(default_path).replace(".csv", ".html"))
        model.set_metadata_prop("fifo_data_path", str(default_path))
        log.info(f"Stored FIFO dataframe to {default_path}.")
    return model


class Simulation:
    """Manage simulation (runs) in FINN. Upon instance creation, the simulation will be built.
    Simulations should inherit from this class and expand for their specific needs.

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
        self.model = self.model.transform(BuildSimulation(fpgapart, clk_ns, functional_sim))
        self.binaries: dict[int, Path] = {i: sim_binaries[i] for i in range(len(sim_binaries))}
        match simulation_type:
            case SimulationType.NODE_BASED_CONNECTED:
                self.binaries = {
                    i: self.binaries[i] / "LayerSimulationBackend" for i in self.binaries.keys()
                }
            case SimulationType.NODE_BASED_ISOLATED:
                self.binaries = {
                    i: self.binaries[i] / "IsolatedSimulationBackend" for i in self.binaries.keys()
                }
            case _:
                raise FINNInternalError(f"Unsupported simulation type: {simulation_type}")

        errors = []
        for binary in self.binaries.values():
            if not binary.exists():
                errors.append(f"Binary {binary} does not exist! Please rerun BuildSimulation!")
        if len(errors) > 0:
            raise FINNInternalError("Errors occurred: \n" + "\n\t".join(errors))

    def simulate(self) -> Any:
        raise NotImplementedError("Call simulate() on subclasses.")


class ApplyFIFOSizes(Transformation):
    """Apply a FIFO sizing configuration to the model.
    If FIFOs already exist the step is skipped."""

    def __init__(
        self,
        cfg: DataflowBuildConfig,
        fifo_config: Path | None = None,
        max_qsrl_depth: int = 256,
        vivado_ram_style: str = "block",
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

        self.fifo_depths: FIFODepthConfig = []
        with self.path.open() as f:
            self.fifo_depths = cast("FIFODepthConfig", json.load(f))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply FIFO Simulation Depths to the model."""
        if len(list(filter(lambda node: "StreamingFIFO" in node.op_type, model.graph.node))) > 0:
            log.warning(
                "It seems that StreamingFIFOs have already "
                "been inserted into the graph. Skipping insertion of FIFOs."
            )
            return model, False

        if len(model.graph.node) != len(self.fifo_depths):
            raise FINNUserError(
                "There are no StreamingFIFOs in the graph, yet the number "
                "of nodes and number of FIFO sizes differ. There may be "
                "unaccounted for nodes that have not been part of the FIFO "
                "simulation. Consider re-running simulation directly before "
                "applying the FIFO sizes. It might also be that your model "
                "or config is outdated, in which case it is recommended to "
                "re-run the entire flow from start to finish."
            )

        # FIFO sizes are set as the maximum of outFIFODepth and inFIFODepth of the successor node
        # Only set the outFIFODepth, because setting both is redundant as inFIFODepth defaults to 0.
        # Remove all in/outFIFODepths in model for clean slate
        graph = model.graph
        for node in graph.node:
            predecessors = model.find_direct_predecessors(node)
            successors = model.find_direct_successors(node)
            n = getCustomOp(node)
            if n is not None:
                if predecessors is not None:
                    n.set_nodeattr("inFIFODepths", [0] * len(predecessors))
                if successors is not None:
                    n.set_nodeattr("outFIFODepths", [0] * len(successors))

        # Set new outFIFODepths according to config
        graph = model.graph
        node_ind = -1
        for first_node in graph.node:
            node_ind += 1
            n0 = getCustomOp(first_node)
            if n0 is None:
                raise FINNInternalError(
                    f"Node {first_node.name} does not have a custom op instance."
                    " This is required for FIFO insertion."
                )
            if first_node.name != self.fifo_depths[node_ind]["node"]:
                raise FINNInternalError(
                    f"Node name {first_node.name} does not match expected name "
                    f"{self.fifo_depths[node_ind]['node']} at index {node_ind}. "
                    "This may be due to a mismatch between the model and the config, "
                    "or due to changes in the model after the simulation was run. "
                    "Consider re-running the entire flow from start to finish."
                )
            fifos = cast("list[int]", (self.fifo_depths[node_ind]["depths"]))
            n0.set_nodeattr("outFIFODepths", fifos)

        # Insert the FIFOs into the model
        model = model.transform(InsertFIFO(True, self.max_qsrl_depth, self.vivado_ram_style))

        model = model.transform(GiveUniqueNodeNames())
        model: ModelWrapper = model.transform(GiveReadableTensorNames())
        model = model.transform(SpecializeLayers(self.cfg._resolve_fpga_part()))  # noqa
        model = model.transform(GiveUniqueNodeNames())
        model: ModelWrapper = model.transform(GiveReadableTensorNames())

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
        if inserted_fifo_count < int(0.4 * float(len(model.graph.node))):
            log.warning(
                "The number of inserted FIFOs makes up less than 40%"
                " of the total number of nodes in the model. This could "
                "point to a potential error."
            )

        return model, False
