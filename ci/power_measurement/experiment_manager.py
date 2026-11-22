"""Experiment manager for measurement workflows."""

# This is used as a workaround because there is some bug with multiprocessing and h5py
import h5py  # noqa
import inspect
import json
import multiprocessing
import os
import sys
import time
import traceback
from finn_plus_driver import get_driver_class
from measurement_util import PAF, Recorder
from pynq import PL

_T0 = time.monotonic()


def em_log(msg):
    """Print an experiment manager log line prefixed with the seconds elapsed since import.

    The board trace carries no timestamps, so these stamps are the only way to attribute the
    per-run measurement time (overlay load, warmup, experiment, report writing).
    """
    print(f"[EM +{time.monotonic() - _T0:6.1f}s] {msg}", flush=True)


def em_reset_clock():
    """Restart the elapsed-time reference of em_log (called once per deployment package)."""
    global _T0
    _T0 = time.monotonic()


# PAF instances per board name: the power analysis library initialises the board's sensors on
# construction, which a long-lived manager (see serve()) should pay once per process
_PAF_CACHE = {}


class ExperimentManager:
    """Manages multiple experiments."""

    # Information Important for ExperimentManager
    #   driver_type     -> Driver Info
    def __init__(self, config_path, working_dir):
        """Initialize experiment manager with configuration."""
        # TODO: add default config
        self._experiments = []
        config = None
        with open(config_path, "r") as fp_json:
            config = json.load(fp_json)

        self._driver_info = config["driver_information"]
        self._experiment_info = config["experiment_information"]

        # The build flow places the external/runtime-writable weight files next to
        # settings.json. The driver's default for runtime_weight_dir is a path relative to the
        # working directory, which is the CI checkout here, so resolve it explicitly.
        self._driver_info.setdefault(
            "runtime_weight_dir",
            os.path.join(os.path.dirname(os.path.abspath(config_path)), "runtime_weights"),
        )

        self._experiment_info["global"]["bitfile_name"] = os.path.join(
            working_dir, self._experiment_info["global"]["bitfile_name"]
        )
        self._experiment_info["global"]["report_path"] = os.path.join(
            working_dir, self._experiment_info["global"]["report_path"]
        )

        # The driver is installed as the finn-plus-driver package on the board, so the
        # deployment package only ships accelerator-specific artifacts.
        driver_type = self._driver_info["driver_type"]
        self._driver_class = get_driver_class(driver_type)

        # ex_config includes:
        # experiment_information/global
        # experiment (e.g first experiment)
        # driver_info
        # driver class instance
        for ex_config in self._experiment_info["experiments"]:
            for gk in self._experiment_info["global"].keys():
                if gk not in ex_config.keys():
                    ex_config[gk] = self._experiment_info["global"][gk]
            ex_config["driver_type"] = self._driver_class
            ex_config["driver_info"] = self._driver_info
            self._experiments.append(Experiment(**ex_config))

        # This is not referenced but sets the _paf global variable,
        # which is used by the other classes
        board = self._experiment_info.get("global").get("PAF").get("board", None)
        if board not in _PAF_CACHE:
            _PAF_CACHE[board] = PAF(board)
        self._paf = _PAF_CACHE[board]

    def start_experiments(self):
        """Run the experiments one after another and return the exit code (0 = all passed).

        Experiments with identical driver parameters (same bitstream and settings) share one
        driver instance, so the bitstream is downloaded and its metadata parsed once per
        deployment package instead of once per experiment.
        """
        exit_code = 0
        shared_driver = None
        shared_params = None
        for experiment in self._experiments:
            reuse = shared_driver is not None and experiment.params_dict == shared_params
            e = experiment.start_experiment(shared_driver=shared_driver if reuse else None)
            if not reuse and experiment._driver_inst is not None:
                shared_driver = experiment._driver_inst
                shared_params = experiment.params_dict
            if e != 0:
                exit_code = 1
                em_log(f"EXPERIMENT {experiment._name} EXITED WITH ERROR: {e}")
        em_log("ALL EXPERIMENTS DONE")
        return exit_code


class Experiment:
    """Represents a single experiment."""

    def __init__(self, **ex_config):
        """Initialize experiment with configuration parameters."""
        # Experiment Meta Information (e.g. warmup)
        self._functions = ex_config.get("functions", [])
        self._name = ex_config.get("name", "")
        self._rails = ex_config.get("PAF").get("rails", [])
        self._sensors = ex_config.get("PAF").get("sensors", [])
        self._power_supply_ip = ex_config.get("PAF").get("power_supply_ip", None)
        self._warmup = ex_config.get("warmup", 0)
        self._iterations = ex_config.get("num_runs", 1)
        self._timeout = ex_config.get("timeout", None)
        self._bitstream_path = ex_config.get("bitstream_path", None)
        self._report_path = ex_config.get("report_path", None)
        self._record_warmup = ex_config.get("PAF").get("record_warmup", False)
        # Sampling interval of the power recorder in seconds (default 1 s): short experiments
        # keep their number of samples by sampling faster
        self._interval = float(ex_config.get("PAF").get("interval", 1.0))
        # The recordings are always saved as json; the xlsx export (nothing in CI reads it,
        # and its first use costs ~10 s of imports on the board) is opt-in
        self._save_xlsx = bool(ex_config.get("PAF").get("save_xlsx", False))
        self._driver_inst = None

        self._driver_class = ex_config.get("driver_type")
        # Inspect Class functions to get the args
        init_signature = inspect.signature(self._driver_class.__init__)
        insert_kwargs = False
        params_dict = {}
        for param in init_signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                insert_kwargs = True
                continue
            if param.name in ex_config:
                # Get value from config
                params_dict[param.name] = ex_config[param.name]
                del ex_config[param.name]
            elif param.name in ex_config["driver_info"]:
                params_dict[param.name] = ex_config["driver_info"][param.name]
            else:
                # Choose default value
                if param.default is not inspect.Parameter.empty:
                    params_dict[param.name] = param.default
                else:
                    raise ValueError(f"Variable {param.name} has no (default) value!!")

        self.insert_kwargs = insert_kwargs
        self.params_dict = params_dict
        self.ex_config = ex_config

    def load_experiment_overlay(self):
        """Load and initialize the experiment overlay."""
        sys.setrecursionlimit(10000)  # PYNQ cache seems to crash otherwise for large designs
        PL.reset()  # See https://github.com/Xilinx/PYNQ/issues/1409

        if self.insert_kwargs:
            self._driver_inst = self._driver_class(**self.params_dict, **self.ex_config)
        else:
            self._driver_inst = self._driver_class(**self.params_dict)

    def start_experiment(self, shared_driver=None):
        """Main method to start an experiment.

        Before an experiment is executed it is initialized and then run
        with the specified configuration. ``shared_driver`` is a driver instance of a
        previous experiment with the same parameters; it is reused instead of loading the
        overlay again.
        """
        em_log(f"SETTING UP EXPERIMENT: {self._name}")
        em_log(f"MONITORING RAILS: {self._rails}")
        em_log(f"MONITORING SENSORS: {self._sensors}")
        if shared_driver is not None:
            self._driver_inst = shared_driver
            em_log(f"REUSING LOADED OVERLAY FOR EXPERIMENT: {self._name}")
        else:
            self.load_experiment_overlay()
            em_log(f"OVERLAY LOADED FOR EXPERIMENT: {self._name}")
        if self._power_supply_ip is not None:
            em_log(f"POWER SUPPLY IP: {self._power_supply_ip}")

        experiment_path = os.path.join(self._report_path, self._name.replace(" ", "_"))

        self._recorder = Recorder(
            self._rails,
            self._sensors,
            self._power_supply_ip,
            interval=self._interval,
            omit_current=True,
            omit_voltage=True,
        )
        em_log(f"STARTING EXPERIMENT: {self._name}")

        for i in range(1, self._iterations + 1):
            iteration_path = os.path.join(experiment_path, f"exp_itr_{i}")
            os.makedirs(iteration_path, exist_ok=True)
            if self._record_warmup:
                self._recorder.start()
                em_log("RECORDING WARMUP")

            for function in self._functions:
                name = function["function_name"]
                kwargs = function.get("kwargs", {})
                kwargs["report_dir"] = iteration_path

                em_log(f"STARTING ITERATION {i} OF FUNCTION {name}, KWARGS= {kwargs}")
                func = getattr(self._driver_inst, name)

                if self._timeout is None:
                    # execute asynchronously to implement proper warmup
                    sys.stdout.flush()
                    p = multiprocessing.Process(target=func, kwargs=kwargs)
                    p.start()
                else:
                    print("[EM]ERROR: Timeout not supported")
                    return 1

                time.sleep(self._warmup)

                if not self._record_warmup and not self._recorder.is_running():
                    em_log(f"STARTING RECORDING AFTER {self._warmup} s OF WARMUP")
                    self._recorder.start()

                # wait on DUT to finish
                p.join()
                em_log(f"FINISHED ITERATION {i} OF FUNCTION {name}")

            # stop & reset recorder and save results
            self._recorder.stop()
            em_log(f"RECORDER STOPPED FOR ITERATION {i} OF EXPERIMENT: {self._name}")
            if self._save_xlsx:
                self._recorder.save_dfs_to_xlsx(experiment_path, f"{self._name}_run_{i}")
            self._recorder.save_dfs_to_json(experiment_path, f"{self._name}_run_{i}")
            self._recorder.reset()
            em_log(f"RESULTS SAVED FOR ITERATION {i} OF EXPERIMENT: {self._name}")

            if (
                p.exitcode != 0
            ):  # If the process exit code is not 0 return with nonzero. This cancels the experiment
                return p.exitcode

        return 0


RUN_FINISHED_MARKER = "EM_RUN_FINISHED"


def run_package(json_path, working_dir):
    """Measure one deployment package and return its exit code (an exception counts as 1)."""
    em_reset_clock()
    try:
        return ExperimentManager(os.path.abspath(json_path), working_dir).start_experiments()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        return 1


def serve():
    """Measure the packages handed over on stdin, one ``<settings.json>\t<working dir>`` per line.

    After each package a line ``EM_RUN_FINISHED <exit code>`` is printed; the process ends at
    EOF. ci/measure.py uses this mode so that the interpreter start, the PYNQ import and the
    board setup are paid once per job instead of once per package.
    """
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        json_path, _, working_dir = line.partition("\t")
        code = run_package(json_path, working_dir)
        print(f"{RUN_FINISHED_MARKER} {code}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--serve":
        serve()
    elif len(sys.argv) == 3:
        sys.exit(run_package(sys.argv[1], sys.argv[2]))
    else:
        em_log("ERROR: Provide settings.json path and working directory as argument, or --serve")
        sys.exit(1)
