try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

# disable common verilator warnings that should be harmless but commonly occur
# in large quantities for Vivado HLS-generated verilog code
commonVerilatorArgs = [
    "-Wno-STMTDLY",
    "-Wno-PINMISSING",
    "-Wno-IMPLICIT",
    "-Wno-WIDTH",
    "-Wno-COMBDLY",
    "-Wno-WIDTHCONCAT",
    "-Wno-UNPACKED",
    "-Wno-TIMESCALEMOD",
    "-Wno-MODDUP",
    "-Wno-CASEINCOMPLETE",
    "-Wno-LITENDIAN",
]


def checkForVerilator():
    """Checks if PyVerilator is installed"""

    if PyVerilator is None:
        raise ImportError("Installation of PyVerilator is required.")


def buildPyVerilator(
    verilog_files,
    verilog_path=[],
    build_dir=None,
    json_data=None,
    gen_only=False,
    trace_depth=2,
    top_module_name=None,
    auto_eval=True,
    read_internal_signals=False,
    extra_args=[],
):
    """Wrapper for PyVerilator which includes args to silence common warnings"""

    return PyVerilator.build(
        verilog_files,
        build_dir=build_dir,
        verilog_path=verilog_path,
        trace_depth=trace_depth,
        top_module_name=top_module_name,
        json_data=None,
        gen_only=False,
        auto_eval=auto_eval,
        read_internal_signals=read_internal_signals,
        extra_args=extra_args + commonVerilatorArgs,
    )
