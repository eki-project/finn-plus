from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import Counter
from pathlib import Path

def inspect_onnx(filename: Path) -> None:
    console = Console()
    model = ModelWrapper(str(filename))
    s = f"[bold cyan]Model: [/bold cyan][bold blue]{filename.name}[/bold blue]\n"
    nodes = model.graph.node
    if len(nodes) < 2:
        console.print("[red]Model has less than 2 nodes[/red]!")
        
    s += f"[cyan]Number layers:[/cyan] {len(nodes)}\n"
    s += f"[cyan]Input node: [/cyan]{nodes[0].name}"
    s += f" - Inputs: {nodes[0].input}\n"
    s += f"[cyan]Output node: [/cyan]{nodes[-1].name}"
    s += f" - Outputs: {nodes[-1].output}\n"

    if "global_in" in nodes[0].input:
        s += "[cyan]Model input: [/cyan]"
        shape = model.get_tensor_shape("global_in")
        dtype = model.get_tensor_datatype("global_in")
        s += f"{shape} [ {dtype} ]\n"
    if "global_out" in nodes[-1].output:
        s += "[cyan]Model output: [/cyan]"
        shape = model.get_tensor_shape("global_out")
        dtype = model.get_tensor_datatype("global_out")
        s += f"{shape} [ {dtype} ]"
    console.print(Panel(s))

    table = Table()
    table.add_column("Op_type")
    table.add_column("Occurences")
    counted = Counter([n.op_type for n in nodes])
    for optype, count in counted.most_common():
        table.add_row(f"[cyan]{optype}[/cyan]", str(count))
    console.print(table)