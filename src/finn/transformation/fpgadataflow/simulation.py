"""Manages the Simulation superclass as well as general simulation related transforms."""

import json
import pandas as pd
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation, SimulationType
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx.onnx_ml_pb2 import NodeProto

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

FIFODepthConfig: TypeAlias = dict[int, dict[str, str | list[int]]]


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
    # TODO: Check if all layers are accounted for
    if len(data.index) != len(model.graph.node):
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
            merged.sort_values(sort_on)
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
