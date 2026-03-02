"""
Base class for FINN benchmarking framework.

This module provides the foundational `bench` class for running automated benchmarks
of FINN dataflow builds. It wraps the existing FINN builder and handles configuration
management, build setup, and artifact collection.

Classes:
    bench: Main benchmarking class that orchestrates the complete build flow
"""

import glob
import os
import shutil
import yaml
from shutil import copy as shcopy
from shutil import copytree

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.util import delete_dir_contents
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import alveo_default_platform, alveo_part_map, part_map
from finn.util.logging import log
from finn.util.settings import get_settings


class bench:
    """
    Base class for FINN benchmarking operations.

    This class provides the foundational framework for running automated benchmarks
    of FINN dataflow builds. It manages the complete lifecycle from configuration
    setup to build execution and artifact collection. Simple flows (also referred to
    as Designs-under-Tests (DUTs) in this context) can use this base class directly, while
    complex flows can subclass and override specific methods as needed.

    Attributes:
        report_dir (str): Directory where build reports are stored
        output_dict (dict): Collection of additional metrics produced by this infrastructure
    """

    def __init__(self, params, task_id, run_id, work_dir, artifacts_dir, save_dir, debug=True):
        """
        Initialize a new benchmark instance that manages a single FINN build.

        Args:
            params (dict): Parameters for the FINN builder and the bench instance itself
            task_id: Identifier for the SLURM task (job array index) or 0 if not using SLURM
            run_id: Unique identifier for this run instance
            work_dir (str): Working directory for build files
            artifacts_dir (str): Directory where output artifacts will be saved
            save_dir (str): Directory where additional (debug) artifacts will be saved
            debug (bool, optional): Enable debug mode for additional artifact collection.
                                  Defaults to True.

        The constructor performs several initialization tasks:
        - Sets up default board and timing configurations
        - Configures shell flow type based on board selection
        - Loads DUT-specific configuration from YAML files
        - Initializes artifact collection mechanisms
        - Prepares build directories and clears previous build artifacts
        """
        super().__init__()
        self._params = params
        self._task_id = task_id
        self._run_id = run_id
        self._work_dir = work_dir
        self._artifacts_dir = artifacts_dir
        self._save_dir = save_dir
        self._debug = debug

        # Setup some basic global default configuration
        # TODO: clean up or remove these attributes
        if "synth_clk_period_ns" in params:
            self._clock_period_ns = params["synth_clk_period_ns"]
        else:
            self._clock_period_ns = 10
            self._params["synth_clk_period_ns"] = self._clock_period_ns

        # TODO: do not allow multiple targets in a single bench job due to measurement?
        if "board" in params:
            self._board = params["board"]
        else:
            self._board = "RFSoC2x2"
            self._params["board"] = self._board

        if "part" in params:
            self._part = params["part"]
        elif self._board in part_map:
            self._part = part_map[self._board]
        else:
            raise Exception("No part specified for board %s" % self._board)

        if self._board in alveo_part_map:
            self._params["shell_flow_type"] = build_cfg.ShellFlowType.VITIS_ALVEO
            self._params["vitis_platform"] = alveo_default_platform[self._board]
        else:
            self._params["shell_flow_type"] = build_cfg.ShellFlowType.VIVADO_ZYNQ

        # Do best afford in mapping DUT to a dataset
        if "validation_dataset" in self._params:
            pass
        elif self._params["dut"] == "bnn-pynq":
            log.warning(
                "DUT bnn-pynq selected. Selecting mnist as the validation dataset. "
                "This might be incorrect. If so configure a validation_dataset in the config"
            )
            self._params["validation_dataset"] = "mnist"
        elif self._params["dut"] == "vgg10":
            self._params["validation_dataset"] = "radioml"
        elif self._params["dut"] in ["mobilenetv1", "resnet50"]:
            self._params["validation_dataset"] = "imagenet"
        elif self._params["dut"] == "cybsec":
            self._params["validation_dataset"] = "unswnb15"
        else:
            # TODO implement for gtsrb, kws, transformer, synthetic_nonlinear (?), mvau (?)
            log.warning(
                "No dataset available for the selected DUT. Configure manually if possible."
            )

        # Load custom (= non build_dataflow_config) parameters from topology-specific .yml
        custom_params = [
            "model_dir",  # used to setup onnx/npy input
            "model_path",  # used to setup onnx/npy input
            # model-gen parameters, such as seed, simd, pe, etc.
            # TODO: separate these more cleanly from builder options
        ]

        if "experiments_config" in params:
            self.experiments_config = params["experiments_config"]
        else:
            # Set default experiment config if not explicitly defined as absolute or relative path
            # TODO: this assumes we are running from the repo root, where ci/ is available
            if "live_fifo_sizing" in params and params["live_fifo_sizing"] is True:
                # Default experiment config for FIFO-Sizing
                self.experiments_config = os.path.join(
                    "ci", "experiments", "fifosizing_default.json"
                )
            else:
                # Default experiment config for normal builds
                # TODO: Switch to default.json to run instr. + DMA validation exp. by default
                self.experiments_config = os.path.join(
                    "ci", "experiments", "instrument_only_default.json"
                )

        dut_yaml_name = self._params["dut"] + ".yml"
        dut_path = os.path.join(os.path.dirname(__file__), "dut", dut_yaml_name)
        if os.path.isfile(dut_path):
            with open(dut_path, "r") as f:
                dut_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            for key in dut_cfg:
                if key in custom_params:
                    self._params[key] = dut_cfg[key]

        # Clear FINN tmp build dir before every run
        print("Clearing FINN BUILD DIR ahead of run")
        delete_dir_contents(get_settings().finn_build_dir)

        # Initialize dictionary to collect all benchmark results
        # TODO: remove completely or only use for meta data,
        # actual results go into run-specific .json files within /report
        self.output_dict = {}

        # Inputs (e.g., ONNX model, golden I/O pair, folding config, etc.)
        self._build_inputs = {}

        # Collect tuples of (name, source path, archive?) to save as pipeline artifacts
        self._artifacts_collection = []

        # Collect tuples of (name, source path, archive?) to save as local artifacts
        self._local_artifacts_collection = []
        if self._debug:
            # Save entire FINN_BUILD_DIR
            # TODO: add option to only save upon error/exception
            self._local_artifacts_collection.append(
                ("debug_finn_tmp", get_settings().finn_build_dir, True)
            )

        # SETUP
        # Use a temporary dir for buildflow-related files (next to FINN_BUILD_DIR)
        # Ensure it exists but is empty (clear potential artifacts from previous runs)
        tmp_buildflow_dir = os.path.join(self._work_dir, "buildflow")
        os.makedirs(tmp_buildflow_dir, exist_ok=True)
        delete_dir_contents(tmp_buildflow_dir)
        self._build_inputs["build_dir"] = os.path.join(
            tmp_buildflow_dir, "build_output"
        )  # TODO remove in favor of self.build_dir
        self._build_dir = os.path.join(tmp_buildflow_dir, "build_output")
        self.report_dir = os.path.join(self._build_dir, "report")
        os.makedirs(self.report_dir, exist_ok=True)

        # Save full build dir as local artifact
        self._local_artifacts_collection.append(("build_output", self._build_dir, False))
        # Save reports and deployment package as pipeline artifacts
        self._artifacts_collection.append(("reports", self.report_dir, False))
        self._artifacts_collection.append(
            ("reports", os.path.join(self._build_dir, "build_dataflow.log"), False)
        )
        self._artifacts_collection.append(("deploy", os.path.join(self._build_dir, "deploy"), True))

    def _save_artifact(self, target_path, source_path, archive=False):
        """
        Save a single artifact from source to target location.

        Args:
            target_path (str): Destination path where artifact will be saved
            source_path (str): Source path of the artifact to save
            archive (bool, optional): If True, create a ZIP archive of the source.
                                    Defaults to False.

        This method handles both files and directories:
        - For directories: copies recursively or creates ZIP archive
        - For files: copies to target directory
        - Automatically creates parent directories as needed
        """
        if os.path.isdir(source_path):
            if archive:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.make_archive(target_path, "zip", source_path)
            else:
                os.makedirs(target_path, exist_ok=True)
                copytree(source_path, target_path, dirs_exist_ok=True)
        elif os.path.isfile(source_path):
            os.makedirs(target_path, exist_ok=True)
            shcopy(source_path, target_path)

    def save_artifacts_collection(self):
        """
        Save all collected pipeline artifacts.

        This method should be called upon successful or failed completion of a run.
        It processes all artifacts in the artifacts_collection list and saves them
        to the pipeline artifacts directory structure.

        Pipeline artifacts typically include:
        - Build reports and logs
        - Deployment packages
        """
        for name, source_path, archive in self._artifacts_collection:
            target_path = os.path.join(
                self._artifacts_dir, "runs_output", "run_%d" % (self._run_id), name
            )
            self._save_artifact(target_path, source_path, archive)

    def save_local_artifacts_collection(self):
        """
        Save all collected local artifacts for debugging.

        This method should be called upon successful or failed completion of a run.
        It processes all artifacts in the local_artifacts_collection list and saves
        them to the local save directory.

        Local artifacts typically include:
        - Complete build output directories
        - FINN build directory contents (when debug=True)
        """
        for name, source_path, archive in self._local_artifacts_collection:
            target_path = os.path.join(self._save_dir, name, "run_%d" % (self._run_id))
            self._save_artifact(target_path, source_path, archive)

    def _step_export_onnx(self):
        """
        Export or generate ONNX model for benchmarking.

        This method must be implemented by subclasses to provide the ONNX model
        that will be processed by the FINN build flow.

        Returns:
            str: Status indicator. Return "skipped" if no model can be generated
                 for the given parameters, which will cause the benchmark run
                 to be skipped.

        Note:
            This is an abstract method that must be overridden by concrete
            benchmark implementations.
        """
        pass

    def _step_build_setup(self):
        """
        Initialize the DataflowBuildConfig for this benchmark.

        This method can be overridden by subclasses if the setup is too complex
        for YAML definition. The default implementation loads configuration from
        a YAML file named after the DUT.

        Returns:
            DataflowBuildConfig: Initialized build configuration object

        Raises:
            Exception: If no DUT-specific YAML build definition is found

        The YAML file should be located at: benchmarking/dut/{dut_name}.yml
        where {dut_name} is the value of params["dut"].
        """
        dut_yaml_name = self._params["dut"] + ".yml"
        dut_path = os.path.join(os.path.dirname(__file__), "dut", dut_yaml_name)
        if os.path.isfile(dut_path):
            with open(dut_path, "r") as f:
                return DataflowBuildConfig.from_yaml(f)
        else:
            raise Exception("No DUT-specific YAML build definition found")

    def run(self):
        """
        Execute the benchmark run.

        This method defaults to running the complete FINN build flow but may be
        overridden by subclasses to implement custom benchmark sequences.

        Returns:
            str: Status of the benchmark run. Returns "skipped" if the benchmark
                 cannot be executed with the given parameters.
        """
        return self._steps_full_build_flow()

    def _step_parse_builder_output(self, build_dir):
        """
        Parse and analyze the output from the FINN builder.

        Args:
            build_dir (str): Path to the build output directory

        This method analyzes the build results and extracts verification status
        information. Currently focuses on checking verification step success by
        examining output files in the verification_output directory.

        The results are stored in self.output_dict for later analysis.

        TODO: Output results as .json or integrate as a new build step
        """
        if os.path.exists(os.path.join(build_dir, "verification_output")):
            # Collect all verification output filenames
            outputs = glob.glob(os.path.join(build_dir, "verification_output/*.npy"))
            # Extract the verification status for each verification output by matching
            # to the SUCCESS string contained in the filename
            status = all([out.split("_")[-1].split(".")[0] == "SUCCESS" for out in outputs])

            # Construct a dictionary reporting the verification status as string
            self.output_dict["builder_verification"] = {
                "verification": {True: "success", False: "fail"}[status]
            }
            # TODO: mark job as failed if verification fails?

    def _steps_full_build_flow(self):
        """
        Execute the complete FINN dataflow build sequence.

        This method implements the default step sequence for benchmarking a full
        FINN builder flow, including:

        1. **Model Creation/Import**: Load or generate ONNX model
        2. **Build Setup**: Configure DataflowBuildConfig with parameters
        3. **Build Execution**: Run the FINN dataflow build pipeline
        4. **Analysis**: Parse and collect build results

        Returns:
            str: "skipped" if the benchmark cannot be executed, otherwise None

        The method handles three model input scenarios:
        - model_dir: Pre-existing ONNX model with verification I/O pairs
        - model_path: Path to existing ONNX model file
        - Generated: ONNX model created by _step_export_onnx()

        Configuration management includes:
        - Loading base configuration from DUT YAML file
        - Setting global defaults for optimization and debugging
        - Merging run-specific parameters
        - Environment setup for build execution
        """
        if "model_dir" in self._params:
            # input ONNX model and verification input/output pairs are provided
            model_dir = self._params["model_dir"]
            self._build_inputs["onnx_path"] = os.path.join(model_dir, "model.onnx")
            self._build_inputs["input_npy_path"] = os.path.join(model_dir, "inp.npy")
            self._build_inputs["output_npy_path"] = os.path.join(model_dir, "out.npy")
        elif "model_path" in self._params:
            self._build_inputs["onnx_path"] = self._params["model_path"]
        else:
            # input ONNX model (+ optional I/O pair for verification) will be generated
            self._build_inputs["onnx_path"] = os.path.join(
                self._build_inputs["build_dir"], "model_export.onnx"
            )
            if self._step_export_onnx(self._build_inputs["onnx_path"]) == "skipped":
                # microbenchmarks might skip because no model can be generated for given params
                return "skipped"

        # BUILD SETUP
        # Initialize from YAML (default) or custom script (if dedicated subclass is defined)
        cfg = self._step_build_setup()

        # Set some global defaults (could still be overwritten by run-specific YAML)
        cfg.output_dir = self._build_inputs["build_dir"]
        # enable extra performance optimizations (physopt)
        # TODO: check OMX synth strategy again!
        cfg.vitis_opt_strategy = build_cfg.VitisOptStrategy.PERFORMANCE_BEST
        cfg.verbose = True
        cfg.console_log_level = build_cfg.LogLevel.ERROR
        cfg.enable_build_pdb_debug = False
        cfg.enable_exception_snapshots = True
        # cfg.stitched_ip_gen_dcp = False # only needed for further manual integration
        cfg.split_large_fifos = True
        cfg.save_intermediate_models = True  # Save the intermediate model graphs
        cfg.verify_save_full_context = True  # Output full context dump for verification steps
        cfg.enable_instrumentation = True
        # rtlsim_use_vivado_comps # TODO ?
        # cfg.default_swg_exception
        # cfg.large_fifo_mem_style

        cfg.experiments_config_path = self.experiments_config

        # Set verification i/o paths if available
        if "input_npy_path" in self._build_inputs and "output_npy_path" in self._build_inputs:
            cfg.verify_input_npy = self._build_inputs["input_npy_path"]
            cfg.verify_expected_output_npy = self._build_inputs["output_npy_path"]

        # Overwrite build config settings with run-specific parameters
        # Filter to only valid DataflowBuildConfig attributes to avoid errors
        valid_params = {k: v for k, v in self._params.items() if hasattr(cfg, k)}

        # Separate params into those that can go through from_dict and those that are None
        params_for_from_dict = {}
        params_with_none = {}

        for k, v in valid_params.items():
            if v == "None":
                # Convert string "None" to actual None
                params_with_none[k] = None
            elif v is None:
                # Explicit None value - set directly to override cfg defaults
                params_with_none[k] = None
            else:
                # Regular value - use from_dict for proper validation and enum conversion
                params_for_from_dict[k] = v

        # TODO: warn/error if there are unrecognized options set?

        # Apply non-None values through from_dict for validation and enum conversion
        if params_for_from_dict:
            updated_cfg = DataflowBuildConfig.from_dict(params_for_from_dict)
            for param_key in params_for_from_dict.keys():
                setattr(cfg, param_key, getattr(updated_cfg, param_key))

        # Apply None values directly to override existing cfg values
        for param_key, param_value in params_with_none.items():
            setattr(cfg, param_key, param_value)

        # disable verification if live FIFO-sizing is on
        if cfg.live_fifo_sizing:
            cfg.verify_steps = None

        # Default of 1M cycles is insufficient for MetaFi (6M) and RN-50 (2.5M)
        # TODO: make configurable or set on pipeline level?
        os.environ["LIVENESS_THRESHOLD"] = "10000000"

        # BUILD
        build.build_dataflow_cfg(self._build_inputs["onnx_path"], cfg)

        # ANALYSIS
        self._step_parse_builder_output(self._build_inputs["build_dir"])
