"""Command line interface for the FINN PYNQ runtime driver."""

import click
import json

from finn_plus_driver import get_driver_class


def parse_kv(ctx, self, value):
    """Parse key-value pairs from CLI arguments."""
    result = {}
    for item in value:
        if len(item) != 2:
            print(item)
            raise click.UsageError(
                "Items must be in form: key=val TYPE. "
                'With datatypes ["Str", "Int", "Bool", "Float"] being supported'
            )
        if item[0].count("=") != 1:
            raise click.BadParameter("Items must be key=value")
        k, v = item[0].split("=", 1)

        data_type = item[1]
        if data_type == "Str":
            v = v
        elif data_type == "Int":
            v = int(v)
        elif data_type == "Bool":
            # Is always True except for v == False
            v = not (v == "False")
        elif data_type == "Float":
            v = float(v)
        else:
            raise click.BadParameter(
                f'Only datatypes ["Str", "Int", "Bool", "Float"] '
                f"are supported. Used datatype: {data_type}"
            )

        result[k] = v
    return result


@click.command(
    "Example: finn-plus-driver -b ../bitfile/finn-accel.bit "
    "-s ./settings.json -f experiment_instrumentation "
    "-ck seed=42 Int -fk runtime=10 Int "
    "-fk report_dir='./report_dir/' Str"
)
@click.option("--bitfile_name", "-b", help="Path to the Bitstream")
@click.option("--settings", "-s", help="Path to the settings.json")
@click.option("--function", "-f", help="Function to be executed")
@click.option(
    "--ckwarg",
    "-ck",
    multiple=True,
    callback=parse_kv,
    nargs=2,
    help=("Keyword argument for the class instance: ... -ck key1=val1 TYPE -ck key2=val2 TYPE"),
)
@click.option(
    "--fkwarg",
    "-fk",
    multiple=True,
    callback=parse_kv,
    nargs=2,
    help=("Keyword argument for the called function: ... -fk key1=val1 TYPE -fk key2=val2 TYPE"),
)
def driver_cli(bitfile_name, settings, function, ckwarg, fkwarg):
    """CLI tool to instantiate driver and execute functions.

    Instantiates a driver class and executes a member function.
    The instantiation implicitly loads a bitstream to the FPGA.
    Requires FINN generated bitstream file and settings.json.
    Driver class is inferred from settings.json, while the called
    member function must be chosen via the function option.
    Kwargs for class instantiation or function call can be input
    via --ckwarg or --fkwarg options respectively.
    Class Kwargs take precedence over settings.json Kwargs.
    """
    with open(settings, encoding="utf-8") as f:
        driver_settings = json.load(f)["driver_information"]

    if ckwarg is None:
        ckwarg = {}
    if fkwarg is None:
        fkwarg = {}

    driver_type = driver_settings["driver_type"]
    input_kwargs = {
        **driver_settings,
        **ckwarg,
    }  # ckwarg has precedence when a key conflict happens

    cla = get_driver_class(driver_type)
    inst = cla(bitfile_name, **input_kwargs)
    func = getattr(inst, function)
    print(func(**fkwarg))
