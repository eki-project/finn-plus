from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import qonnx.util
from functools import reduce
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from collections import Counter
from pathlib import Path

def _create_sdp_tree(tree_name: str, model: ModelWrapper, cycle_estimates: dict, color_estimate: callable, ignore_fifos, collapse_consecutive_fifos, display_cycle_estimates, ignore_sdp_prefix) -> Tree:
    """Inspect a modelwrapper and return a list of strings representing branches to add to the main tree"""
    console = Console()
    branch = Tree(f"[bold purple]{tree_name}[/bold purple]")
    have_seen_fifo = False
    for nn in model.graph.node:
        # Consider FIFO nodes
        if "StreamingFIFO" in nn.op_type:
            if ignore_fifos:
                continue
            if collapse_consecutive_fifos and not have_seen_fifo:
                have_seen_fifo = True
                branch.add(f"[grey74]FIFOs[/grey74]")
                continue
            elif collapse_consecutive_fifos and have_seen_fifo:
                continue
        elif have_seen_fifo:
            have_seen_fifo = False 
        
        # Replace name prefix
        if ignore_sdp_prefix:
            printname = nn.name.replace(tree_name + "_", "")
        else:
            printname = nn.name
        
        # Construct string for cycle estimate
        cycle_est = cycle_estimates[tree_name][nn.name]
        if display_cycle_estimates and cycle_est is not None and cycle_est > 0:
            estimate_print = f" --------- Estimate: " + color_estimate(cycle_est) + " cyc"
        else:
            estimate_print = ""
        
        # Add the actual branch
        match nn.op_type:
            case "StreamingFIFO_rtl" | "StreamingFIFO_hls":
                branch.add(f"[grey74]{printname}[/grey74]{estimate_print}")
            case "FMPadding_rtl" | "FMPadding_hls":
                branch.add(f"[bold green]{printname}[/bold green]{estimate_print}")
            case "MVAU_hls" | "MVAU_rtl" | "VVAU_rtl" | "VVAU_hls":
                branch.add(f"[bold blue]{printname}[/bold blue]{estimate_print}")
            case "DuplicateStreams_hls" | "DuplicateStreams_rtl" | "AddStreams_hls" | "AddStreams_rtl":
                branch.add(f"[bold wheat4]{printname}[/bold wheat4]{estimate_print}")
            case "ConvolutionInputGenerator_rtl" | "ConvolutionInputGenerator_hls" | "Pool_hls" | "Pool_rtl":
                branch.add(f"[bold dark_orange]{printname}[/bold dark_orange]{estimate_print}")
            case "StreamingDataWidthConverter_hls" |  "StreamingDataWidthConverter_rtl":
                branch.add(f"[bold dark_red]{printname}[/bold dark_red]{estimate_print}")
            case "Thresholding_rtl" | "Thresholding_hls":
                branch.add(f"[bold green1]{printname}[/bold green1]{estimate_print}")
            case "ChannelwiseOp_hls" | "ChannelwiseOp_rtl" | "LabelSelect_hls" | "LabelSelect_rtl":
                branch.add(f"[bold gold1]{printname}[/bold gold1]{estimate_print}")
            case "DownSampler_hls" | "DownSampler_rtl":
                branch.add(f"[bold cyan]{printname}[/bold cyan]{estimate_print}")
            case _:
                console.print(f"Unknown optype: {nn.op_type}")
                branch.add(f"[grey53]{printname}[/grey53]{estimate_print}")
    return branch


def _try_find_sdp_model(n: "ONNXNode", parent_path: Path) -> Path | None:
    # TODO: Make smart
    try:
        p = Path(getCustomOp(n).get_nodeattr("model"))
    except:
        return None
    alternative_path = (parent_path / "kernel_partitions" / p.name).absolute()
    if p.exists():
        return p
    elif alternative_path.exists():
        return alternative_path
    else:
        return None


def _create_cycle_estimate_dict(model: ModelWrapper) -> dict[str, int]:
    """Collect cycle estimates for this model"""
    d = {}
    for node in model.graph.node:
        try:
            d[node.name] = getCustomOp(node).get_nodeattr("cycles_estimate")
        except:
            d[node.name] = 0
    return d


def _generate_estimate_color_function(estimates: list[int]) -> callable:
    """Generate a function to color cycle estimates appropiately"""
    quad = max(estimates) // 4
    def color_func(value):
        if value <= quad:
            return f"[bold green3]{value}[/bold green3]"
        elif value <= quad*2:
            return f"[bold yellow3]{value}[/bold yellow3]"
        elif value <= quad*3:
            return f"[bold dark_orange3]{value}[/bold dark_orange3]"
        else:
            return f"[bold red1]{value}[/bold red1]"
    return color_func



def inspect_onnx(filename: Path, ignore_sdp_prefix: bool = True, display_cycle_estimates: bool = True, collapse_consecutive_fifos: bool = True, ignore_fifos: bool = True) -> None:
    """
    ignore_sdp_prefix: Replace StreamingDataflowPartition_XY with blank
    display_cycle_estimates: Display cycle estimates for all nodes that have it available. Color by how bad the latency is
    collapse_consecutive_fifos: Collapse names of consecutive FIFOs into one
    ignore_fifos: Dont print anything for fifos in the tree. Ignores "collapse_consecutive_fifos"
    """
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
        s += f"{shape} [ {dtype} ]\n"
    
    # Find the largest tensor
    if all(["finn" in n.domain for n in nodes]):
        tensors = []
        for node in nodes:
            tensors += node.input
            tensors += node.output
        tensors = list(set(tensors))
        largest_shape = 0
        largest_index = 0
        multiple_largest = 0
        for i in range(len(tensors)):
            shape = model.get_tensor_shape(tensors[i])
            bitwidth = model.get_tensor_datatype(tensors[i]).bitwidth()
            current_elem = reduce(lambda a,b:a*b, shape, 1) * bitwidth 
            if current_elem > largest_shape:
                largest_shape = current_elem
                largest_index = i
                multiple_largest = 0
            elif current_elem == largest_shape:
                multiple_largest += 1
        s += f"[cyan]Largest tensor: [/cyan]{tensors[largest_index]} [ {model.get_tensor_shape(tensors[largest_index])} - {model.get_tensor_datatype(tensors[largest_index])} -> {largest_shape} ({multiple_largest} others like this) ]"


    # Are we in an SDP or outside?
    contains_sdps = any([x.op_type == "StreamingDataflowPartition" for x in nodes])

    # Collect cycle estimates
    cycle_estimates = {}
    if not contains_sdps:
        cycle_estimates["Model"] = _create_cycle_estimate_dict(model)
        color_estimate = _generate_estimate_color_function(list(cycle_estimates["Model"].values()))
    else:
        all_estimates = []
        for n in nodes:
            if n.op_type == "StreamingDataflowPartition":
                submodel_path = _try_find_sdp_model(n, filename.parent)
                if submodel_path is None:
                    s += f"[bold dark_orange]Could not locate SDP model {n.name}\n"
                else:
                    cycle_estimates[n.name] = _create_cycle_estimate_dict(ModelWrapper(str(submodel_path)))
                    all_estimates += list(cycle_estimates[n.name].values())
        color_estimate = _generate_estimate_color_function(all_estimates) 


    # Display tree of models
    tree = Tree("Dataflow Partitions")
    if not contains_sdps:
        tree.add(
            _create_sdp_tree(
                "Model", 
                model, 
                cycle_estimates, 
                color_estimate, 
                ignore_fifos, 
                collapse_consecutive_fifos, 
                display_cycle_estimates, 
                ignore_sdp_prefix
            )
        )
    for n in nodes:
        if n.op_type == "StreamingDataflowPartition":
            submodel_path = _try_find_sdp_model(n, filename.parent)
            if submodel_path is None:
                s += f"[bold dark_orange]Could not locate SDP model {n.name}\n"
                continue
            else:
                submodel = ModelWrapper(str(submodel_path))
                tree.add(
                    _create_sdp_tree(
                        n.name, 
                        submodel, 
                        cycle_estimates, 
                        color_estimate, 
                        ignore_fifos, 
                        collapse_consecutive_fifos, 
                        display_cycle_estimates, 
                        ignore_sdp_prefix
                    )
                )

    console.print(Panel(s))
    table = Table()
    table.add_column("Op_type")
    table.add_column("Occurences")
    counted = Counter([n.op_type for n in nodes])
    for optype, count in counted.most_common():
        table.add_row(f"[cyan]{optype}[/cyan]", str(count))
    console.print(table)
    console.print(tree)