"""Experiment manager for measurement workflows."""

# This is used as a workaround because there is some bug with multiprocessing and h5py
import h5py  # noqa
import importlib.util
import inspect
import json
import multiprocessing
import os
import sys
import time
from measurement_util import PAF, Recorder
from pynq import PL


class ExperimentManager:
    """Manages multiple experiments."""

    # Information Important for ExperimentManager
    #   driver (path)   -> Experiment Info
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

        self._experiment_info["global"]["driver"] = os.path.join(
            working_dir, self._experiment_info["global"]["driver"]
        )
        self._experiment_info["global"]["bitfile_name"] = os.path.join(
            working_dir, self._experiment_info["global"]["bitfile_name"]
        )
        self._experiment_info["global"]["report_path"] = os.path.join(
            working_dir, self._experiment_info["global"]["report_path"]
        )

        # add driver path to path for experiments
        self._driver = self._experiment_info["global"]["driver"]
        if self._driver not in sys.path:
            print("[EM] Adding driver to path")
            sys.path.insert(0, self._driver)
        else:
            print("[EM] Driver already part of path")

        # setup driver module
        spec = importlib.util.spec_from_file_location("driver", self._driver)

        # Add the directory of the driver to sys.path
        module_dir = os.path.dirname(spec.origin)
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        self._driver_module = importlib.util.module_from_spec(spec)
        sys.modules["driver"] = self._driver_module
        spec.loader.exec_module(self._driver_module)

        driver_type = self._driver_info["driver_type"]
        self._driver_class = getattr(sys.modules["driver"], driver_type)

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
        self._paf = PAF(self._experiment_info.get("global").get("PAF").get("board", None))

    def start_experiments(self):
        """Main method to commence experimentation.

        Each experiement specified in the configuratin gets executed after another.
        """
        exit_code = 0
        for experiment in self._experiments:
            e = experiment.start_experiment()
            if e != 0:
                exit_code = 1
                print(f"[EM] EXPERIMENT {experiment._name} EXITED WITH ERROR: {e}")
        sys.exit(exit_code)


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

    def start_experiment(self):
        """Main method to start an experiment.

        Before an experiment is executed it is initialized and then run
        with the specified configuration.
        """
        print(f"[EM] SETTING UP EXPERIMENT: {self._name}")
        print(f"[EM] MONITORING RAILS: {self._rails}")
        print(f"[EM] MONITORING SENSORS: {self._sensors}")
        self.load_experiment_overlay()
        if self._power_supply_ip is not None:
            print(f"[EM] POWER SUPPLY IP: {self._power_supply_ip}")

        experiment_path = os.path.join(self._report_path, self._name.replace(" ", "_"))

        self._recorder = Recorder(
            self._rails, self._sensors, self._power_supply_ip, omit_current=True, omit_voltage=True
        )
        print(f"[EM] STARTING EXPERIMENT: {self._name}")

        for i in range(1, self._iterations + 1):
            iteration_path = os.path.join(experiment_path, f"exp_itr_{i}")
            os.makedirs(iteration_path, exist_ok=True)
            if self._record_warmup:
                self._recorder.start()
                print("Record warmup")

            for function in self._functions:
                name = function["function_name"]
                kwargs = function.get("kwargs", {})
                kwargs["report_dir"] = iteration_path

                print(f"[EM] STARTING ITERATION {i} OF FUNCTION {name}, KWARGS= {kwargs}")
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
                    print(f"[EM] STARTING RECORDING AFTER {self._warmup} s OF WARMUP", flush=True)
                    self._recorder.start()

                # wait on DUT to finish
                p.join()

            # stop & reset recorder and save results
            self._recorder.stop()
            self._recorder.save_dfs_to_xlsx(experiment_path, f"{self._name}_run_{i}")
            self._recorder.save_dfs_to_json(experiment_path, f"{self._name}_run_{i}")
            self._recorder.reset()

            if (
                p.exitcode != 0
            ):  # If the process exit code is not 0 return with nonzero. This cancels the experiment
                return p.exitcode

        return 0


if __name__ == "__main__":
    if len(sys.argv) == 3:
        json_path = os.path.abspath(sys.argv[1])
        working_dir = sys.argv[2]

        ex_flow = ExperimentManager(json_path, working_dir)
        ex_flow.start_experiments()
    else:
        print("[EM] ERROR: Provide settings.json path and working directory as argument")
        sys.exit(1)
