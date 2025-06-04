# Copyright 2025 Paderborn University
# Originally authored by Lucas Reuter
# Modified by Felix Jentzsch

import importlib.util
import IPython
import json
import multiprocessing
import os
import pandas as pd
import plotly.graph_objects as go
import sys
import time
import vxi11
from cffi import FFI
from plotly.subplots import make_subplots
from threading import Event, Thread

# flake8: noqa

_ffi = None
_paf = None


class PAF:
    """
    Main class used for power analysis and examination.

    The power analysis framework (PAF) encapsulates the basic CFFI interfce neded to gather power readouts aswell as
    additional information from supported development boards. This class serves as the main way to acces onboard information
    and handles the initialization of the needed onboard infrastructre.
    """

    _rails = {}
    _sensors = {}
    _lib = None
    supported_boards = ("zcu102", "rfsoc2x2")

    def __init__(self, board):
        """Initialize the PAF.
        Creates the CFFI interface by defining the desired C-Header and importing the corresponding .so file for the development
        board. As part of the initialization the initialize() function of the development board gets called to ensure proper
        board setup.
        """
        global _ffi
        global _paf

        if board not in PAF.supported_boards:
            raise ValueError(
                f"Board '{board}' not supported yet. Please choose one of the supported boards: {PAF.supported_boards}. Alternatively implement the needed dirver."
            )

        _ffi = FFI()
        _ffi.cdef(
            """struct sensor {
            char* name;
            int id;
            char* unit;
            double value;
            void (*update_value)(int id);
                };"""
        )

        _ffi.cdef(
            """struct rail {
            char* rail_name;
            int id;
            double voltage;
            double current;
            double power;
            void (*update_values)(int id);
                };"""
        )

        _ffi.cdef("""int initialize();""")
        _ffi.cdef("""struct rail** get_rails();""")
        _ffi.cdef("""int get_num_rails(); """)
        _ffi.cdef("""struct sensor** get_sensors(); """)
        _ffi.cdef("""int get_num_sensors(); """)

        framework_dirname = os.path.dirname(os.path.abspath(__file__))
        board_driver_path = os.path.abspath(f"{framework_dirname}/boards/{board}/{board}.so")
        if os.path.isfile(board_driver_path):
            PAF._lib = _ffi.dlopen(board_driver_path)
        else:
            raise FileNotFoundError(
                f"Board driver not found at {board_driver_path}. Please ensure the file exists by running 'make shared' in the driver folder."
            )

        PAF._lib.initialize()
        self._init_rails()
        self._init_sensors()
        _paf = self

    def _init_rails(self):
        """Initilize the PAF rail dictionary with rail names as keys and rail C-Structs as values."""
        if len(PAF._rails) == 0 and not PAF._lib.get_num_rails() == 0:
            for rail in _ffi.unpack(PAF._lib.get_rails(), PAF._lib.get_num_rails()):
                rail_obj = Rail(rail)
                PAF._rails[rail_obj.name] = rail_obj

    def _init_sensors(self):
        """Initialize the PAF sensor dictionary with sensor names as keys and sensor C-Structs as values."""
        if len(PAF._sensors) == 0 and not PAF._lib.get_num_sensors() == 0:
            for sensor in _ffi.unpack(PAF._lib.get_sensors(), PAF._lib.get_num_sensors()):
                sensor_obj = Sensor(sensor)
                PAF._sensors[sensor_obj.name] = sensor_obj

    def get_rails(self):
        """Get the current rails.

        Returns
        -------
        dict
            Dictionary containing rail names as keys an Rail objects as values.
        """
        return PAF._rails

    def get_sensors(self):
        """Get the current sensors.

        Returns
        -------
        dict
            Dictionary containing sensor names as keys an Sensor objects as values.
        """
        return PAF._sensors


class Rail:
    def __init__(self, rail):
        """Initialize a python object wrapping the CFFI-Rail-Object.

        Parameters
        ----------
        rail
            CFFI-Rail-Object for the corresponding rail
        """
        self._rail = rail
        self._livevis = None

    @property
    def name(self):
        """Gets the name of the rail.

        Returns
        -------
        string
            Name of the rail
        """
        global _ffi
        return _ffi.string(self._rail.rail_name).decode("utf-8")

    @property
    def voltage(self):
        """Gets the most recent voltage reading of the rail.

        Returns
        -------
        float
            Most recent voltage
        """
        return self._rail.voltage

    @property
    def current(self):
        """Gets the most recent current reading of the rail.

        Returns
        -------
        float
            Most recent current
        """
        return self._rail.current

    @property
    def power(self):
        """Gets the most recent power reading of the rail.

        Returns
        -------
        float
            Most recent power
        """
        return self._rail.power

    def update(self):
        """Updates voltage, current, and power values with the most recent ones.

        If a LiveVisualization is attached to the Rail the LiveVisualization gets
        updated aswell.
        """
        self._rail.update_values(self._rail.id)
        if self._livevis is not None:
            self._livevis.update_plots([self.voltage, self.current, self.power])

    def get_livevis(self):
        """Get a LiveVisualization representing the current values of the rail.

        If the rail has no current LiveVisualization associated a new one will be created.

        Returns
        -------
        LiveVisualization
            A LiveVisualization offering a visual representation of the rail readouts.
        """
        if self._livevis is None:
            self._livevis = LiveVisualization([self.name, "voltage", "current", "power"])
        return self._livevis


class Sensor:
    def __init__(self, sensor):
        """Initialize a python object wrapping the CFFI-Sensor-Object.

        Parameters
        ----------
        rail
            CFFI-Sensor-Object for the corresponding sensor.
        """
        self._sensor = sensor

    @property
    def name(self):
        """Gets the name of the sensor.

        Returns
        -------
        string
            Name of the sensor.
        """
        global _ffi
        return _ffi.string(self._sensor.name).decode("utf-8")

    @property
    def value(self):
        """Gets the most recent sensor value.

        Returns
        -------
        float
            Most recent sensor value.
        """
        return self._sensor.value

    @property
    def unit(self):
        """Gets the unit of the sensor values.

        Returns
        -------
        string
            Unit of sensor values.
        """
        global _ffi
        return _ffi.string(self._sensor.unit).decode("utf-8")

    def update(self):
        self._sensor.update_value(self._sensor.id)


class Recorder:
    """A Recorder records multiple rails and sensors at the same time."""

    def __init__(
        self,
        rails=[],
        sensors=[],
        power_supply_ip=None,
        interval=1.0,
        num_datapoints=-1,
        omit_voltage=False,
        omit_current=False,
        omit_group_elements=False,
        output_dir=".",
    ):
        """Initialize the recorder with desired settings.

        Parameters
        ----------
        rails : list
            String list of rail names to be recorded. Also allows groups.
        sensors : list
            String list of sensor names to be recorded.
        power_supply_ip : string
            String containing the ip for the power supply
        interval : float
            Desired upate interval of the recorder.
        num_datapoints : int
            Number of datapoints to be caputred. If left unchanged the recorder will run indefinitely until stopped, otherwise
            datacapturing is stopped when num_datapoints have been aquired.
        omit_voltages : bool
            If true the voltages of rails will not be recorded.
        omit_current: bool
            If true the currents of rails will not be recorded.
        omit_group_elements : bool
            If true rails of a group will be omitted and only the total rail groups power will be recorded.
        """
        self._index = 0
        self._omit_voltage = omit_voltage
        self._omit_current = omit_current
        self._omit_group_elements = omit_group_elements
        self._interval = interval
        self._rails = []
        self._sensors = []
        self._power_supply = PowerSupply(power_supply_ip) if power_supply_ip is not None else None
        self._num_datapoints = num_datapoints
        self._df_rails, self._df_sensors = self._prepare_df(rails, sensors, self._power_supply)
        self._stopped = Event()
        self._thread = Thread(
            target=self._thread_record_data, args=[self._rails, self._sensors, self._power_supply]
        )
        self._running = False
        self._output_dir = output_dir

    def _prepare_rail_group(self, group):
        """Helper to create column names for grouped rails.

        Parameters
        ----------
        group : list
            List representing the group name and the rails to be grouped. Group name is the first entry, followed by
            a list of rail names (e.g.["GROUP_NAME",["RAIL1", "RAIL2"]]).

        Returns
        -------
        list
            List of column names that need to be added to the dataframe.
        """
        group_name = group[0]
        group_rails = group[1]
        columns = []
        rail_group = []

        for rail in group_rails:
            rail_group.append(_paf._rails[rail])
            if not self._omit_group_elements:
                columns = columns + self._prepare_rail(rail, False)
        self._rails.append(
            rail_group
        )  # FIXME no need to append here. Remove this and add_to_rails from _prepare rail, just let _prepare_rail add the rails
        columns.append(group_name + "_power")
        return columns

    # TODO add_to_rails can be removed when _prepare_rail_group gets changed
    # add_to_rails essentially distinguishes if _prepare_rails has been called by _prepare_rail_group
    def _prepare_rail(self, rail, add_to_rails=True):
        """Helper to create column names for rails.

        The corresponding column names will be omitted if their omit flag has been set.

        Parameters
        ----------
        rail : string
            Name of the rail that needs to be added.
        add_to_rails : bool
            Signals if the rail has to be added to the interal _rails list. This list aggregates
            all rails that need to be updated when recording.

        Returns
        -------
        list
            List of column names that need to be added to the dataframe.
        """
        columns = []
        if not self._omit_voltage:
            columns.append(rail + "_voltage")
        if not self._omit_current:
            columns.append(rail + "_current")
        columns.append(rail + "_power")

        if add_to_rails:
            self._rails.append(_paf._rails[rail])
        return columns

    def _prepare_sensor(self, sensor):
        """Helper to create column names for sensors.

        Returns
        -------
        columns : list
            List of column names that need to be added to the dataframe.
        """
        columns = []
        columns.append(sensor + "_val")
        self._sensors.append(_paf._sensors[sensor])
        return columns

    def _prepare_df(self, rails, sensors, power_supply):
        """Helper to fill the recorder dataframes with their column names.

        Parameters
        ----------
        rails : list
            String list of rail names.
        sensors : list
            String list of sensor names.
        power_supply : PowerSupply
            PowerSupply object of the connected power supply.

        Returns
        -------
        tuple
            Tuple containing two dataframes. The first dataframe contains the header for the appropriate rails.
            The second the header for the appropriate sensors.
        """
        # prepare dataframe for rails
        rails_columns = []

        for element in rails:
            # check if the element in rails list is a list aswell
            # this indicates grouping of a rail
            if isinstance(element, list):
                rails_columns = rails_columns + self._prepare_rail_group(element)
            else:
                rails_columns = rails_columns + self._prepare_rail(element)

        rails_columns.append("total_power")

        if power_supply is not None:
            if not self._omit_voltage:
                rails_columns.append("power_supply_voltage")
            if not self._omit_current:
                rails_columns.append("power_supply_current")
            rails_columns.append("power_supply_power")

        df_rails = pd.DataFrame(columns=rails_columns)

        # prepare dataframe for sensors
        sensors_columns = []
        for sensor in sensors:
            sensors_columns = sensors_columns + self._prepare_sensor(sensor)
        df_sensors = pd.DataFrame(columns=sensors_columns)

        return (df_rails, df_sensors)

    def start(self):
        """Starts the recorder."""
        self._thread.start()
        self._running = True

    def stop(self):
        """Stops the reocrder."""
        self._stopped.set()
        self._thread.join()
        self._running = False

    def get_dfs(self):
        """Gets the current recorder dataframes for rails and sensors.

        Returns
        -------
        list
            List containing the rail dataframe and sensor dataframe.
        """
        return [self._df_rails.loc[:], self._df_sensors.loc[:]]

    def is_running(self):
        """Check if the recorder is currently running.

        Returns
        -------
        bool
            True if the recorder is running. False otherwise.
        """
        return self._running

    def reset(self):
        """Resets the recorder to its initial state while keeping dataframe columns.

        This allows the recorder to be reused again.
        """

        # stop recorder when it's running before resetting
        if not self._stopped.is_set():
            self.stop()

        # reset dataframes but keep columns
        self._df_rails = self._df_rails.iloc[0:0]
        self._df_sensors = self._df_sensors.iloc[0:0]

        # reset stopped flag
        self._stopped.clear()

        # reset index count
        self._index = 0

        # reset running status
        self._running = False

        # reset thread
        self._thread = Thread(
            target=self._thread_record_data, args=[self._rails, self._sensors, self._power_supply]
        )

    def save_dfs_to_xlsx(self, title):
        """Saves the recorder dataframes as .xlsx.

        Parameters
        ----------
        title : string
            Name of the xlsx file to export to.
        """
        os.makedirs(self._output_dir, exist_ok=True)  # ensure output directory exists
        with pd.ExcelWriter(os.path.join(self._output_dir, f"{title}.xlsx")) as writer:
            self._df_rails.to_excel(writer, sheet_name="rails")
            self._df_sensors.to_excel(writer, sheet_name="sensors")

    def _thread_record_data(self, rails, sensors, power_supply):
        """This method is the main recording loop for the recorder thread.

        Parameters
        ----------
        rails : list
            List of rail objects that are recorded. IMPORTANT: Note that this expect as list of rail
            object NOT a list of rail names.
        sensors : list
            List of sensor objects that are recorded. NOTE: See rails
        power_supply : PowerSupply
            PowerSupply object of the connected power supply.
        """
        while not self._stopped.is_set():
            if self._num_datapoints > 0:
                for self._index in range(self._num_datapoints):
                    self._record_data(rails, sensors, power_supply)
                    time.sleep(self._interval)

                # stop data recording after num_datapoints acquired
                self._stopped.set()
            else:
                self._record_data(rails, sensors, power_supply)
                time.sleep(self._interval)

    def _update_group(self, rail_group):
        """Update a group of rails according to the set omit flags.

        Parameters
        ----------
        rail_group : list
            List of Rail objects which are part of the group that will be measured.

        Returns
        -------
        list
            Measurement values for the group which can be added to the recordings.
        """
        measurements = []

        group_power = 0.0
        for rail in rail_group:
            rail.update()
            if not self._omit_group_elements:
                if not self._omit_voltage:
                    measurements.append(rail.voltage)
                if not self._omit_current:
                    measurements.append(rail.current)
                measurements.append(rail.power)
            group_power += rail.power
        measurements.append(group_power)
        return measurements

    def _record_data(self, rails, sensors, power_supply):
        """This method handles how the recorder dataframes are updated.

        This is achiebed by utilizing the corresponding rail and sensors objects and using their
        update methods.

        Parameters
        ----------
        rails : list
            String list of rail names to record.
        sensors : list
            String list of sensors names to record.
        power_supply : PowerSupply
            PowerSupply object of the connected power supply.
        """

        if len(rails) > 0:
            rail_measurements = []
            total_power = 0.0
            for element in rails:
                if isinstance(element, list):
                    group_measurements = self._update_group(element)
                    total_power += group_measurements[-1]
                    rail_measurements = rail_measurements + group_measurements
                else:
                    element.update()
                    if not self._omit_voltage:
                        rail_measurements.append(element.voltage)
                    if not self._omit_current:
                        rail_measurements.append(element.current)
                    rail_measurements.append(element.power)
                    total_power += element.power
            rail_measurements.append(total_power)

            if power_supply is not None:
                if not self._omit_voltage:
                    rail_measurements.append(power_supply.measure_voltage("CH1"))
                if not self._omit_current:
                    rail_measurements.append(power_supply.measure_current("CH1"))
                rail_measurements.append(power_supply.measure_power("CH1"))
            self._df_rails.loc[self._index, :] = rail_measurements

        if len(sensors) > 0:
            sensor_measurements = []
            for sensor in sensors:
                sensor.update()
                sensor_measurements.append(sensor.value)
            self._df_sensors.loc[self._index, :] = sensor_measurements

        self._index += 1


class PowerSupply:
    """PowerSupply encapsulates basic capabilities to communicate with a power supply via VXI11."""

    def __init__(self, ip):
        """Initialize the power supply object

        Parameters
        ----------
        ip : string
            String containing the ip of the power supply.
        """
        self._instrument = vxi11.Instrument(ip)

    def get_channel_settings(self, channel):
        """Returns the current channel settings.

        Parameters
        ----------
        channel : string
            String containing the channel name to request the settings from (e.g. CH1, CH2, etc.).

        Returns
        -------
        string
            The current channel settings summarized in a single string.
        """
        return self._instrument.ask(f":APPL? {channel}")

    def set_channel_settings(self, channel, voltage, current):
        """Sets the channel to the specified voltage and current.

        NOTE: To avoid settings that could damage the hardware, only the currently needed setting is supported.
        This method needs to be changed to support any other settings.

        Parameters
        ----------
        channel : string
            String containing the channel name to apply the settings to(e.g. CH1, CH2, etc.).
        voltage : string
            String containing the desired output voltage the channel will be set to.
        current : string
            String containing the desired output current the channel will be set to.
        """
        if channel == "CH1" and voltage == "12.2" and current == "10":
            self._instrument.write(
                f":OUTP {channel}, OFF"
            )  # turn off output before setting new values
            time.sleep(1)
            self._instrument.write(f":APPL {channel},{voltage},{current}")
            time.sleep(1)
            self._instrument.write(f":OUTP {channel}, ON")  # turn channel on again
        else:
            print(
                "ERROR: Trying to set something other than CH1 to 12.2V @ 10A is not supported!!!"
            )  # TODO change this if other settings should be supported

    def measure_voltage(self, channel):
        """Returns the current output voltage of the channel.

        Parameters
        ----------
        channel : string
            String containing the channel name to measure the current output voltage from.

        Returns
        -------
        float
            Latest channel outpu voltage.
        """
        return float(self._instrument.ask(f":MEAS:VOLT? {channel}"))

    def measure_current(self, channel):
        """Returns the current output voltage of the channel.

        Parameters
        ----------
        channel : string
            String containing the channel name to measure the current output voltage from.

        Returns
        -------
        float
            Latest channel output current.
        """
        return float(self._instrument.ask(f":MEAS:CURR? {channel}"))

    def measure_power(self, channel):
        """Returns the current output power of the channel.

        Parameters
        ----------
        channel : string
            String containing the channel name to measure the current output power from.

        Returns
        -------
        float
            Latest channel output power.
        """
        return float(self._instrument.ask(f":MEAS:POWE? {channel}"))


class LiveVisualization:
    """LiveVisualization bundles the functionality needed to offer a live preview of rail readouts in JupyterLab notebooks."""

    def __init__(self, names):
        """Initializes the LiveVisualization with the required subplots.

        names : list
            String list of the different value names that are visualized. IMPORTANT: The first entry is the rail name which is used to
            set the plot title.
        """
        subplots = self._build_subplots(
            names[1:]
        )  # first entry in names is plot title, the rest are trace names
        self._fig_widget = go.FigureWidget(subplots)
        self._fig_widget.layout.title = names[0]

    def show_plot(self):
        """Shows the LiveVisualization plot.

        NOTE: This method needs to be called in a JupyterLab notebook.
        """
        IPython.display.display(self._fig_widget)

    def _build_subplots(self, names):
        """Helper method to build the needed subplots for the LiveVisualization.

        Parameters
        ----------
        names : list
            List of names for the visualized values.
        """
        sub_plot = make_subplots(rows=1, cols=len(names))
        sub_plot.layout.showlegend = True
        for i in range(len(names)):
            sub_plot.add_scatter(y=[], row=1, col=i + 1, name=names[i])
        return sub_plot

    def update_plots(self, data):
        """Routine to update the plots."""
        for i in range(len(data)):
            self._fig_widget.data[i].y = self._fig_widget.data[i].y + (data[i],)

    def reset_plots(self):
        """Resets the plots to be empty."""
        for plot in self._fig_widget.data:
            plot.y = []


class Experiment:
    """Experiment represents one experiment that is to be conducted.

    This encapsulates the automated configuration of the PAF and required recorder. An Experiment offers
    ways to call custom experimentation code with custom imports, even from local files. This relies on
    a proper configuration to be provided to the Experiment object.
    """

    def __init__(self, config=None, **kwargs):
        """Initalize the experiment with its desired configuration.

        Parameters
        ----------
        config : dict
            A configuration dictionary containing the required parameters for the experiment initialization.
        title : string, optional
            Title of the experiment
        functions : list
            A list of functions that are to be called during the experiment. Each entry of the list specifies
            a single function call from the experiment as follows. A function call is a dictionary containing the following
            keys and values.
            "name" : "The name of the function to call as a string" (e.g. "some_function")
            "args" : A list of arguments to pass to the function
            "kwargs" : A dictionary containing keyword parameters.
        power_supply_ip : string, optional
            A string containing the IP of the powersupply.
        warmup : float, optional
            Time in seconds waited before the functions defined in 'functions' are called. Recording will already commence.
        end_after : float, optional
            Time in seconds after which the experiment will be stopped forcefully. WARNING: Not working properly when conducting
            multiple experiments in sequence.
        iterations : int, optional
            Number of times the experiment shall be executed. Default is 1
        driver : string
            Path to the PYNQ driver.
        rails : list
            List containing the names of the rails to be recorded as strings.
        sensors : list
            List containing the names of the sensors to be recorded as strings.
        bitstream : string
            Path to the bitstream.
        import_paths : list
            List of strings containing path to local imports which might be needed to run the experiment.
        experiment_path : string
            Path to the .py file which contains the definitions for the functions called from 'functions'
        board : string
            Name of the board model. This is needed to choose the appropriate shared object to obtain data from the board.
        record_warmup : bool
            Flag indicating if the warmup period should be recorded or not. Default is False
        """

        if config is not None:
            self._config = config

            self._title = config.get("title", "")
            self._functions = config.get("functions", [])
            self._power_supply_ip = config.get("PAF").get("power_supply_ip", None)
            self._warmup = config.get("warmup", 0)  # warmup is default 0s
            self._end_after = config.get(
                "end_after", None
            )  # end experiment after set seconds, None indicates no timeout
            self._iterations = config.get("iterations", 1)  # number of iterations is default 1
            self._driver = config.get("FINN").get("driver", None)
            self._rails = config.get("PAF").get("rails", [])
            self._sensors = config.get("PAF").get("sensors", [])
            self._bitstream_path = config.get("FINN").get("bitstream", None)
            self._import_paths = config.get("import_paths", [])
            self._experiment_path = config.get("experiment_path", "")
            self._report_path = config.get("report_path", ".")
            self._board = config.get("PAF").get("board", None)
            self._record_warmup = config.get("PAF").get("record_warmup", False)

            self._recorder = None
            self._experiment_module = None
        else:
            # TODO: remove kwargs in favor of json config
            self._title = kwargs.get("title", "")
            self._functions = kwargs.get("functions", [])
            self._power_supply_ip = kwargs.get("power_supply_ip", None)
            self._warmup = kwargs.get("warmup", 0)
            self._end_after = kwargs.get("end_after", None)
            self._iterations = kwargs.get("iterations", 1)
            self._driver = kwargs.get("driver", None)
            self._rails = kwargs.get("rails", [])
            self._sensors = kwargs.get("sensors", [])
            self._bitstream_path = kwargs.get("bitstream", None)
            self._import_paths = kwargs.get("import_paths", [])
            self._experiment_path = kwargs.get("experiment_path", "")
            self._board = kwargs.get("board", None)
            self._record_warmup = kwargs.get("record_warmup", False)

            self._recorder = None
            self._experiment_module = None

            self._config = self._create_config()

        self._paf = PAF(self._config.get("PAF").get("board", None))

    def _create_config(self):
        """Helper method to generate a json config based on the current configuration variables."""

        config_dict = {}

        # construct FINN section of config
        config_dict["FINN"] = {}
        config_dict["FINN"]["driver"] = self._driver
        config_dict["FINN"]["bitstream"] = self._bitstream_path

        # construct PAF section of config
        config_dict["PAF"] = {}
        config_dict["PAF"]["rails"] = self._rails
        config_dict["PAF"]["sensors"] = self._sensors
        config_dict["PAF"]["power_supply_ip"] = self._power_supply_ip
        config_dict["PAF"]["board"] = self._board
        config_dict["PAF"]["record_warmup"] = self._record_warmup

        # construct general section of config
        config_dict["title"] = self._title
        config_dict["import_paths"] = self._import_paths
        config_dict["functions"] = self._functions
        config_dict["experiment_path"] = self._experiment_path

        return config_dict

    # TODO add flag to add wrapper which makes config suitable to be loaded from CLI
    def export_config(self, name):
        """Export the current experiment configuration as a json file.

            This file can be used to either load the configuration at a later point or to serve
            as a starting point to configure an ExperimentFlow.

        Parameters
        ----------
        name : string
            Name of the json configuration file
        """

        with open(f"{name}.json", "w") as json_file:
            json.dump(self._config, json_file, indent=4)

    def _setup_experiment(self):
        """Helper method to setup the experiments environment.

        This includes adding custom paths to the PATH environment variable, adding the FINN driver, and importing
        the experiment as a module that can be interfaced with. The setup also includes the creation of a Recorder
        to aquire measurements.
        """

        # add all paths for  experiment
        # check if path is already in sys.path
        # if so do not add it again and remove from _import_paths
        # to avoid unwanted deletion of this path in _disassemble_experiment()
        for path in self._import_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
            else:
                print(
                    f"Path: {path} already part of sys.path. Removing from experiment import paths"
                )
                self._import_paths.remove(path)

        # add driver path aswell
        if self._driver not in sys.path:
            print("Adding driver to path")
            sys.path.insert(0, self._driver)
        else:
            print("driver already part of path")

        # setup experiment module
        spec = importlib.util.spec_from_file_location("experiment", self._config["experiment_path"])
        self._experiment_module = importlib.util.module_from_spec(spec)
        sys.modules["experiment"] = self._experiment_module
        spec.loader.exec_module(self._experiment_module)

        print(f"rails: {self._rails}\nsensors: {self._sensors}")
        print(f"power_supply_ip {self._power_supply_ip}")
        self._recorder = Recorder(
            self._rails,
            self._sensors,
            self._power_supply_ip,
            omit_current=True,
            omit_voltage=True,
            output_dir=self._report_path,
        )

    def start_experiment(self):
        """Main method to start an experiment.

        Before an experiment is executed it is initialized and then run with the specified configuration.
        """
        self._setup_experiment()

        for i in range(1, self._iterations + 1):
            print("Startring experiment run", i)

            if self._record_warmup:
                self._recorder.start()

            for function in self._config["functions"]:
                name = function["name"]
                args = function.get("args", None)
                kwargs = function.get("kwargs", None)
                kwargs["bitfile"] = self._config["FINN"]["bitstream"]

                print(f"Calling func_name {name} with args: {args}\n kwargs {kwargs}")
                func = getattr(self._experiment_module, name)

                if self._end_after is None:
                    # func(*args, **kwargs)
                    # execute asynchronously to implement proper warmup
                    p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
                    p.start()
                else:
                    print(
                        "ERROR, DO NOT USE THIS FEATURE"
                    )  # TODO: fix for fixed warmup, what if multiple functions are defined?
                    p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
                    p.start()
                    print(f"Experiment will be stopped after end_afer = {self._end_after}s")
                    time.sleep(self._end_after)
                    p.terminate()  # FIXME pynq cannot allocate memory after killing process
                    # os.kill(p.pid, signal.SIGTERM)

                print(f"Warmup for {self._warmup}s before starting measurment...")
                time.sleep(self._warmup)

                if not self._record_warmup and not self._recorder.is_running():
                    self._recorder.start()

                # wait on DUT to finish
                p.join()

            # stop & reset recorder and save results
            self._recorder.stop()
            self._recorder.save_dfs_to_xlsx(f"{self._title}_run_{i}")
            self._recorder.reset()

        self._disassemble_experiment()

    def _disassemble_experiment(self):
        """Helper method to deinitialize an experiment.

        This includes restoring the state of PATH before the experiment
        has been intialized
        """

        for import_path in self._import_paths:
            if import_path in sys.path:
                sys.path.remove(import_path)
        del self._experiment_module
        del sys.modules["experiment"]


# Utilize Experiment Class to implement a series of experiments
class ExperimentFlow:
    """Mutliple Experiments performed in succession are modeled by an ExperimentFlow.

    An ExperimentFlow serves as the point of entry when performing multiple experiments, or when
    calling the paframework from an CLI.
    """

    def __init__(self, json_path):
        """Initializes the ExperimentFlow

        Parameters
        ----------
        json_path : string
            String containing the path to the configuration.
        """
        self._experiments = []
        self._config = None
        with open(json_path, "r") as fp_json:
            self._config = json.load(fp_json)
        self._setup_experiments()

    def _setup_experiments(self):
        """Helper method to setup the experiments described in the configuration."""

        for ex_config in self._config["experiments"]:
            self._append_global(ex_config)
            self._experiments.append(Experiment(ex_config))

    def start_experiments(self):
        """Main method to commence experimentation.

        Each experiement specified in the configuratin gets executed after another.
        """
        for experiment in self._experiments:
            experiment.start_experiment()

    # Checks for a given experient config if the globally defined sections are present or not.
    # If experiment already defines global section leave as is, otherwise append global section to experiment.
    def _append_global(self, experiment_config):
        """Helper method to check if sections from the global configuration need to be appended
        to an experiment configuration.
        """

        for gk in self._config["global"].keys():
            if gk not in experiment_config.keys():
                print(f"Section {gk} not in experiment configuration. Adding section...")
                experiment_config[gk] = self._config["global"][gk]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        json_path = os.path.abspath(sys.argv[1])
        ex_flow = ExperimentFlow(json_path)
        ex_flow.start_experiments()
    else:
        print("Please provide path to experiment json config as argument")
