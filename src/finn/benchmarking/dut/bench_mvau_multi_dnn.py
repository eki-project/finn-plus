"""Benchmark DUT for multi-DNN MVAU configurations."""

import json
import numpy as np
import os
from copy import deepcopy
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.bench_base import bench
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth


class bench_mvau_multi_dnn(bench):
    """Benchmark class for multi-DNN MVAU hardware configurations."""

    def __init__(
        self,
        params: dict,
        task_id: int,
        run_id: int,
        work_dir: str,
        artifacts_dir: str,
        save_dir: str,
        debug: bool = False,
    ) -> None:
        """Initialize benchmark with parameters and output directories."""
        super().__init__(params, task_id, run_id, work_dir, artifacts_dir, save_dir, debug=debug)

    def _make_single_mvau_model(
        self,
        weights: np.ndarray,
        num_input_vectors: list[int],
        pe: int,
        simd: int,
        m: int,
        wdt: BaseDataType,
        idt: BaseDataType,
        odt: BaseDataType,
        thresholds: np.ndarray | None = None,
        tdt: BaseDataType | None = None,
        mem_mode: str = "const",
        ram_style: str = "auto",
        ram_style_thresholds: str = "auto",
        backend: str = "hls",
    ) -> ModelWrapper:
        """Build and return a single MVAU ONNX model with the given parameters."""
        mw = weights.shape[0]
        mh = weights.shape[1]

        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
            export_idt = DataType["BINARY"]
            binary_xnor_mode = 1
        else:
            export_wdt = wdt
            export_idt = idt
            binary_xnor_mode = 0

        inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [*num_input_vectors, mw])
        outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [*num_input_vectors, mh])
        if thresholds is not None:
            no_act = 0
            node_inp_list = ["inp", "weights", "thresh"]
            actval = 0 if odt == DataType["BIPOLAR"] else odt.min()
        else:
            node_inp_list = ["inp", "weights"]
            actval = 0
            no_act = 1

        if backend == "hls":
            customop_name = "MVAU_hls"
            domain = "finn.custom_op.fpgadataflow.hls"
            res_type = "lut"
        elif backend == "rtl":
            customop_name = "MVAU_rtl"
            domain = "finn.custom_op.fpgadataflow.rtl"
            res_type = "dsp"
        else:
            raise ValueError(f"Unsupported backend: {backend} (supported: 'hls', 'rtl')")

        mvau_node = helper.make_node(
            customop_name,
            node_inp_list,
            ["outp"],
            domain=domain,
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            M=m,
            numInputVectors=num_input_vectors,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            ActVal=actval,
            binaryXnorMode=binary_xnor_mode,
            noActivation=no_act,
            resType=res_type,
            mem_mode=mem_mode,
            ram_style=ram_style,
            ram_style_thresholds=ram_style_thresholds,
            runtime_writeable_weights=0,
        )

        graph = helper.make_graph(
            nodes=[mvau_node], name="mvau_graph", inputs=[inp], outputs=[outp]
        )
        model = qonnx_make_model(graph, producer_name="mvau-model")
        model = ModelWrapper(model)

        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", odt)
        model.set_tensor_datatype("weights", wdt)
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer("weights", (weights + 1) / 2)
        else:
            model.set_initializer("weights", weights)
        if thresholds is not None:
            model.set_tensor_datatype("thresh", tdt)
            model.set_initializer("thresh", thresholds)

        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(InferDataTypes())
        return model

    def _apply_sparsity(self, weights: np.ndarray, mw: int, mh: int) -> np.ndarray:
        """Apply random sparsity to the weight matrix."""
        sparsity_amount = self._params.get("sparsity_amount", 0)
        if sparsity_amount == 0:
            return weights
        # NPY002: the benchmark seeds the legacy global RNG explicitly (np.random.seed)
        # to keep weight matrices reproducible across runs; a Generator would not see it.
        idx = np.random.choice(  # noqa: NPY002
            mw * mh, size=int(sparsity_amount * mw * mh), replace=False
        )
        weights = np.reshape(weights, -1)
        weights[idx] = 0.0
        return np.reshape(weights, (mw, mh))

    def _step_export_onnx(self) -> None:
        """Generate and save multi-DNN ONNX models and config."""
        result = self._generate_multi_dnn_models_and_config()
        self._multi_dnn_config_path = result

    def _generate_multi_dnn_models_and_config(self) -> str | None:
        """Create two MVAU submodels and their multi-DNN config JSON; return config path."""
        scenario, mem_mode = self._params["scenario_mem_mode"]
        idt = DataType[self._params["idt"]]
        wdt = DataType[self._params["wdt"]]

        num_input_vectors = self._params["nhw"]
        mw = self._params["mw"]
        mh = self._params["mh"]
        pe, simd = self._params["pe_simd"]
        m = self._params["m"]
        ram_style = self._params["ram_style"]
        ram_style_thr = self._params["ram_style_thr"]
        backend = self._params["backend"]
        output_dict = {}
        if pe > mh or simd > mw:
            print("Invalid pe/simd configuration, skipping")
            return None
        if mw % simd != 0 or mh % pe != 0:
            print("Invalid simd/pe configuration, skipping")
            return None

        output_dict["simd"] = simd
        output_dict["pe"] = pe
        output_dict["sparsity_amount"] = self._params.get("sparsity_amount")
        output_dict["mw"] = mw
        output_dict["mh"] = mh
        output_dict["idt"] = self._params["idt"]
        output_dict["wdt"] = self._params["wdt"]
        output_dict["m"] = m
        output_dict["nhw"] = num_input_vectors
        output_dict["mem_mode"] = mem_mode
        output_dict["ram_style"] = ram_style
        output_dict["backend"] = backend
        output_dict["scenario"] = scenario

        np.random.seed(123456)  # noqa: NPY002  (seeds the legacy global RNG used below)
        weights_a = gen_finn_dt_tensor(wdt, (mw, mh))
        weights_a = self._apply_sparsity(weights_a, mw, mh)

        num_zeros = (weights_a == 0).sum()
        output_dict["zero_weights"] = round(num_zeros / weights_a.size, 2)

        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            odt = DataType["UINT32"]
        else:
            odt = DataType["INT32"]

        model_a = self._make_single_mvau_model(
            weights_a,
            num_input_vectors,
            pe,
            simd,
            m,
            wdt,
            idt,
            odt,
            mem_mode=mem_mode,
            ram_style=ram_style,
            ram_style_thresholds=ram_style_thr,
            backend=backend,
        )
        model_a.graph.name = "mvau_A"
        model_a = model_a.transform(GiveUniqueNodeNames())

        model_b = deepcopy(model_a)
        model_b.graph.name = "mvau_B"

        mvau_node_a = model_a.graph.node[0]
        weight_tensor_name = mvau_node_a.input[1]
        actual_wdt = model_a.get_tensor_datatype(weight_tensor_name)

        np.random.seed(654321)  # noqa: NPY002  (seeds the legacy global RNG used below)
        weights_b = gen_finn_dt_tensor(actual_wdt, (mw, mh))
        weights_b = self._apply_sparsity(weights_b, mw, mh)
        model_b.set_initializer(weight_tensor_name, weights_b)

        model_a_path = str(Path(self._build_dir) / "model_A.onnx")
        model_b_path = str(Path(self._build_dir) / "model_B.onnx")
        model_a.save(model_a_path)
        model_b.save(model_b_path)

        with (Path(self._build_dir) / "report" / "dut_info.json").open("w") as f:
            json.dump(output_dict, f, indent=2)

        cfg_dict = self._create_multi_dnn_config_json(scenario, model_a_path, model_b_path, backend)
        cfg_json_path = str(Path(self._build_dir) / "multi_dnn_config.json")
        with Path(cfg_json_path).open("w") as f:
            json.dump(cfg_dict, f, indent=2)

        return cfg_json_path

    def _create_multi_dnn_config_json(
        self, scenario: int, model_a_path: str, model_b_path: str, backend: str
    ) -> dict:
        """Return a multi-DNN configuration dict for the given scenario."""
        post_collapse_steps = [
            {"step_minimize_bit_width": "Collapsed_Model"},
            {"step_generate_estimate_reports": "Collapsed_Model"},
            {"step_prepare_nodecontainer": "Collapsed_Model"},
            {"step_hw_codegen": "Collapsed_Model"},
            {"step_hw_ipgen": "Collapsed_Model"},
            {"step_create_stitched_ip": "Collapsed_Model"},
            {"step_synthesize_bitfile": "Collapsed_Model"},
            {"step_make_driver": "Collapsed_Model"},
            {"step_deployment_package": "Collapsed_Model"},
        ]
        steps = [
            {"step_apply_multi_dnn": "Multi_DNN_Wrapper"},
            {"step_collapse_multi_dnn": "Multi_DNN_Wrapper"},
            *post_collapse_steps,
        ]

        if scenario == 0:
            generation = {
                "mode": "Parallel",
                "kwargs": {
                    "combine_inputs_channelwise": self._params.get(
                        "combine_inputs_channelwise", True
                    ),
                    "combine_outputs_channelwise": self._params.get(
                        "combine_outputs_channelwise", True
                    ),
                },
            }
            # Insert SIMD-maximization step right after collapse so that the
            # StreamingConcat/Split nodes get proper folding before any downstream
            # step (e.g. estimate reports, codegen) uses their cycle counts.
            collapse_idx = next(i for i, s in enumerate(steps) if "step_collapse_multi_dnn" in s)
            maximize_simd_step = (
                "finn.transformation.multi_dnn.multi_dnn_steps.step_maximize_concat_split_simd"
            )
            steps.insert(collapse_idx + 1, {maximize_simd_step: "Collapsed_Model"})
        elif scenario == 3:
            generation = {
                "mode": "SelectableWeights",
                "kwargs": {"models": ["mvau_A", "mvau_B"]},
            }
        elif scenario == 4:
            mvau_node_name = f"MVAU_{backend}_0"
            # pblock = self._params.get("pblock", "CLOCKREGION_X1Y1:CLOCKREGION_X3Y5")
            generation = {
                "mode": "PartialReconfiguration",
                "kwargs": {
                    "reference_model_name": "mvau_A",
                    "pr_regions": {
                        "pr_mvau_0": {
                            "mvau_A": [mvau_node_name],
                            "mvau_B": [mvau_node_name],
                            # "pblock": pblock,
                        }
                    },
                },
            }
        else:
            raise ValueError(f"Unsupported multi_dnn scenario: {scenario} (supported: 0, 3, 4)")

        return {
            "Submodels": {
                "mvau_A": {"model_path": model_a_path},
                "mvau_B": {"model_path": model_b_path},
            },
            "Steps": steps,
            "Generation": generation,
        }

    def _step_build_setup(self) -> DataflowBuildConfig:
        """Return a default DataflowBuildConfig for the multi-DNN build."""
        cfg = build_cfg.DataflowBuildConfig(
            target_fps=None,
            steps=None,
        )
        return cfg

    def run(self) -> None:
        """Run the multi-DNN build flow."""
        return self._steps_multi_dnn_build_flow()

    def _steps_multi_dnn_build_flow(self) -> None:
        """Execute multi-DNN build flow steps in order."""
        cfg = self._step_build_setup()
        multi_dnn_cfg_path = self._generate_multi_dnn_models_and_config()
        if multi_dnn_cfg_path is None:
            return

        cfg.multi_dnn_config_path = multi_dnn_cfg_path
        cfg.output_dir = self._build_dir
        cfg.vitis_opt_strategy = build_cfg.VitisOptStrategy.PERFORMANCE_BEST
        cfg.verbose = True
        cfg.console_log_level = build_cfg.LogLevel.ERROR
        cfg.enable_build_pdb_debug = False
        cfg.enable_exception_snapshots = True
        cfg.split_large_fifos = True
        cfg.save_intermediate_models = True
        cfg.verify_save_full_context = True
        cfg.enable_instrumentation = True
        cfg.experiments_config_path = self.experiments_config
        valid_params = {
            k: v
            for k, v in self._params.items()
            if hasattr(cfg, k) and k != "multi_dnn_config_path"
        }
        params_for_from_dict = {}
        params_with_none = {}
        for k, v in valid_params.items():
            if v == "None" or v is None:
                params_with_none[k] = None
            else:
                params_for_from_dict[k] = v
        if params_for_from_dict:
            updated_cfg = DataflowBuildConfig.from_dict(params_for_from_dict)
            for pk in params_for_from_dict:
                setattr(cfg, pk, getattr(updated_cfg, pk))
        for pk, pv in params_with_none.items():
            setattr(cfg, pk, pv)
        os.environ["LIVENESS_THRESHOLD"] = "10000000"
        build.build_dataflow_cfg(None, cfg)
        self._step_parse_builder_output(self._build_dir)
