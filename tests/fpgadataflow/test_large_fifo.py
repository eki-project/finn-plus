# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause


import pytest

import json
import torch
from brevitas.export import export_qonnx
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.give_unique_node_names_recursive import GiveUniqueNodeNamesRecursive
from finn.util.basic import make_build_dir, part_map, robust_rmtree
from tests.testing_util.test import get_trained_network_and_ishape

# deep enough to need more than one BRAM/URAM primitive, and not a power of two,
# so it also covers fifo.sv's lo+hi memory space decomposition
DEPTH = 45000


def fetch_test_model(topology, wbits=2, abits=2):
    tmp_output_dir = make_build_dir("build_large_fifo_%s_" % topology)
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir + "/model.onnx"
    export_qonnx(model, torch.randn(ishape), chkpt_name)
    return tmp_output_dir


def make_build_cfg(tmp_output_dir, board, versal, **kwargs):
    return build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        target_fps=10000,
        synth_clk_period_ns=10.0,
        board=None if versal else board,
        fpga_part=part_map[board] if versal else None,
        shell_flow_type=None if versal else build_cfg.ShellFlowType.VIVADO_ZYNQ,
        **kwargs,
    )


def make_fifo_cfg(tmp_output_dir, cfg, depth=DEPTH, ram_style="auto"):
    """FIFO depths are applied from a fifo_config_file keyed by FIFO node name, so
    replay the FIFO insertion of step_set_fifo_depths on the estimate-stage model
    to learn the names and request the same depth/ram_style for all of them."""
    model = ModelWrapper(
        tmp_output_dir + "/intermediate_models/step_generate_estimate_reports.onnx"
    )
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(GiveReadableTensorNames())
    fifo_names = [n.name for n in model.get_nodes_by_op_type("StreamingFIFO_rtl")]
    assert len(fifo_names) > 0
    fifo_cfg = {
        "fifo_depths": {n: depth for n in fifo_names},
        "impl_style": {n: "rtl" for n in fifo_names},
        "ram_style": {n: ram_style for n in fifo_names},
    }
    fifo_config_file = tmp_output_dir + "/fifo_config.json"
    with open(fifo_config_file, "w") as f:
        json.dump(fifo_cfg, f, indent=2)
    return fifo_config_file


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
# "ultra" only on the boards that have URAM. AUP-ZU3_8GB is an xczu3eg, which has no
# URAM, and is left at "auto": fifo.sv would still elaborate the ultra branch at this
# depth, but Vivado drops the attribute (Synth 8-12187) and backs the array with BRAM,
# so asking for it explicitly would test nothing the other two boards do not.
@pytest.mark.parametrize(
    "board, ram_style", [("AUP-ZU3_8GB", "auto"), ("ZCU104", "ultra"), ("VEK280", "ultra")]
)
def test_large_fifo_is_not_split(board, ram_style):
    versal = board in ("VEK280", "VCK190")
    tmp_output_dir = fetch_test_model("tfc")
    # dry run up to the estimate reports to derive the FIFO configuration
    cfg = make_build_cfg(tmp_output_dir, board, versal, stop_step="step_generate_estimate_reports")
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    fifo_config_file = make_fifo_cfg(tmp_output_dir, cfg, DEPTH, ram_style)
    cfg = make_build_cfg(
        tmp_output_dir,
        board,
        versal,
        auto_fifo_depths=False,
        fifo_config_file=Path(fifo_config_file),
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    # every FIFO must build as one instance at exactly the requested depth:
    # no chain of smaller FIFOs, no rounding up to a power of two
    with open(tmp_output_dir + "/report/fifo_sizing.json") as f:
        fifo_info = json.load(f)
    fifos = fifo_info["fifo_depths"]
    assert len(fifos) > 0
    for name, depth in fifos.items():
        assert depth == DEPTH, "%s was split or rounded: %d" % (name, depth)

    robust_rmtree(tmp_output_dir)
