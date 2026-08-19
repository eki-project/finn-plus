# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""ILP-based partitioner for multi-FPGA placement of FINN dataflow graphs."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    from qonnx.core.modelwrapper import ModelWrapper

import networkx as nx
import numpy as np
from mip import BINARY, Model, OptimizationStatus, SearchEmphasis, Var, minimize, xsum
from qonnx.custom_op.registry import getCustomOp

from finn.analysis.fpgadataflow.floorplan_params import floorplan_params
from finn.util.basic import part_map
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.util.platforms import DEFAULT_AVG_CONSTRAINTS, DEFAULT_RES_LIMITS, platforms


class ILP_partitioner:
    """ILP-based partitioner for mapping tasks to FPGAs."""

    def __init__(self) -> None:
        """Initialize the ILP partitioner."""
        super().__init__()
        self.avg_util_constrains = []

    def create_model(
        self,
        task_requirements: list[list[tuple[int, ...]]],
        task_dependencies: list[tuple[int, int]],
        task_dependencies_requirements: list[tuple[int, int]],
        compute_resources: np.ndarray,
        compute_connection_cost: list[list[float]],
        compute_connection_resource: list[list[list[float]]],
        compute_resource_limits: np.ndarray,
        abs_anchors: list[tuple[int, list[int]]] | None = None,
        rel_anchors: list[tuple[int, int]] | None = None,
    ) -> None:
        """Build the ILP model for task-to-compute-node assignment."""
        if abs_anchors is None:
            abs_anchors = []
        if rel_anchors is None:
            rel_anchors = []
        time_init = time.perf_counter()
        model = Model("floorplan")

        # aux lists
        task_nodes = list(range(len(task_requirements)))
        compute_nodes = set(range(len(compute_resources)))
        task_versions = [list(range(len(task_requirements[task]))) for task in task_nodes]

        # binary variables indicating if task_node[j] goes to compute_node[i] or not
        opt_placement = [
            [[model.add_var(var_type=BINARY) for version in task] for task in task_requirements]
            for i in compute_nodes
        ]

        # binary variables indicating if task_node[j] goes
        # from compute_node[o] to compute_node[d] or not
        opt_connection_matrix = [
            [
                [model.add_var(var_type=BINARY) for j in range(len(task_dependencies))]
                for o in compute_nodes
            ]
            for d in compute_nodes
        ]

        # objective function: crossings cost
        model.objective = minimize(
            xsum(
                [
                    opt_connection_matrix[o][d][j] * compute_connection_cost[o][d]
                    for j in range(len(task_dependencies))
                    for o in compute_nodes
                    for d in compute_nodes
                ]
            )
        )

        # constraint 1: link vars opt_placement
        # and opt_connection_matrix to generate valid partition
        for j in range(len(task_dependencies)):
            for o in compute_nodes:
                src_task = task_dependencies[j][0]
                model += xsum(opt_connection_matrix[o][d][j] for d in compute_nodes) == xsum(
                    opt_placement[o][src_task][v] for v in task_versions[src_task]
                )

        for j in range(len(task_dependencies)):
            for d in compute_nodes:
                dst_task = task_dependencies[j][1]
                model += xsum(opt_connection_matrix[o][d][j] for o in compute_nodes) == xsum(
                    opt_placement[d][dst_task][v] for v in task_versions[dst_task]
                )

        # constraint 2: not exceed compute resources
        for i in compute_nodes:
            for r in range(len(compute_resources[0])):
                model += (
                    xsum(
                        [
                            xsum(
                                task_requirements[j][v][r] * opt_placement[i][j][v]
                                for v in task_versions[j]
                            )
                            for j in task_nodes
                        ]
                    )
                    <= (compute_resources * compute_resource_limits).astype(np.int_)[i][r]
                )

        # constraint 3: not exceed connection resources
        for o in compute_nodes:
            for d in compute_nodes:
                for cr in range(len(task_dependencies_requirements[0])):
                    if compute_connection_resource[o][d][cr] >= 0:
                        model += (
                            xsum(
                                opt_connection_matrix[o][d][td]
                                * task_dependencies_requirements[td][cr]
                                for td in range(len(task_dependencies))
                            )
                            <= compute_connection_resource[o][d][cr]
                        )

        # constraint 4: each task is allocated once and only once
        for j in task_nodes:
            sos_vars = [opt_placement[i][j][v] for v in task_versions[j] for i in compute_nodes]
            model += xsum(sos_vars) == 1

        # constraint 5: anchor constrains
        for task, compute_node_list in abs_anchors:
            model += (
                xsum(
                    xsum(opt_placement[i][task][v] for v in task_versions[task])
                    for i in compute_node_list
                )
                == 1
            )

        for task_1, task_2 in rel_anchors:
            for i in compute_nodes:
                model += xsum(opt_placement[i][task_1][v] for v in task_versions[task_1]) == xsum(
                    opt_placement[i][task_2][v] for v in task_versions[task_2]
                )

        self.time_create_model = time.perf_counter() - time_init

        # store useful variables
        self.model = model
        self.opt_placement = opt_placement
        self.opt_connection_matrix = opt_connection_matrix
        self.task_requirements = task_requirements
        self.task_dependencies = task_dependencies
        self.task_dependencies_requirements = task_dependencies_requirements
        self.compute_resources = compute_resources
        self.compute_connection_cost = compute_connection_cost
        self.compute_connection_resource = compute_connection_resource
        self.abs_anchors = abs_anchors
        self.rel_anchors = rel_anchors

    def add_average_of_utilizations_constrain(
        self, resource_numbers: list[int], limit: float
    ) -> None:
        """Implement average utilization constraints for the specified resource types."""
        task_nodes = list(range(len(self.task_requirements)))
        compute_nodes = set(range(len(self.compute_resources)))
        task_versions = [list(range(len(self.task_requirements[task]))) for task in task_nodes]
        for i in compute_nodes:
            self.model += xsum(
                [
                    xsum(
                        [
                            xsum(
                                self.task_requirements[j][v][r] * self.opt_placement[i][j][v]
                                for v in task_versions[j]
                            )
                            for j in task_nodes
                        ]
                    )
                    / self.compute_resources[i][r]
                    for r in resource_numbers
                    if self.compute_resources[i][r] > 0
                ]
            ) <= limit * len(resource_numbers)

        self.avg_util_constrains += [(resource_numbers, limit)]

    def solve_model(
        self,
        emphasis: SearchEmphasis = SearchEmphasis.DEFAULT,
        max_seconds: float = np.inf,
        max_gap: float = 1e-4,
        verbose: bool = False,
    ) -> tuple[OptimizationStatus, float, int]:
        """Solve the ILP model and return solution status, objective value, and solution count."""
        self.model.emphasis = emphasis
        self.model.max_gap = max_gap

        time_init = time.perf_counter()
        self.solution_status = self.model.optimize(max_seconds=max_seconds)
        self.time_solve_model = time.perf_counter() - time_init
        # if fails first increase a little the failing resource limit until MAX_LIMIT,
        # then increase number of SLRs
        if verbose:
            print(
                self.solution_status,
                " | Objective func cost:",
                self.model.objective_value,
                " | Num of solutions: ",
                self.model.num_solutions,
            )

        return (
            self.solution_status,
            self.model.objective_value,
            self.model.num_solutions,
        )

    def get_optimal_placement(
        self,
    ) -> list[list[list[Var]]]:
        """Return optimal placement decision variables."""
        return self.opt_placement

    def get_optimal_connection_matrix(
        self,
    ) -> list[list[list[Var]]]:
        """Return optimal connection matrix decision variables."""
        return self.opt_connection_matrix

    def get_run_times(
        self,
    ) -> tuple[float, float]:
        """Return model creation and solving runtimes in seconds."""
        return self.time_create_model, self.time_solve_model

    def report_best_solution(
        self,
        compute_resources_names: list[str | int] | None = None,
        compute_connection_resource_name: list[str | int] | None = None,
        verbose: bool = True,
    ) -> None:
        """Print best solution including resource utilization and connection stats."""
        if (
            self.solution_status != OptimizationStatus.OPTIMAL
            and self.solution_status != OptimizationStatus.FEASIBLE
        ):
            print("No solution")
            return

        print("Solution:")
        task_nodes = list(range(len(self.opt_placement[0])))
        compute_nodes = list(range(len(self.opt_placement)))
        task_versions = [list(range(len(self.task_requirements[task]))) for task in task_nodes]
        if compute_resources_names is None:
            compute_resources_names = list(range(len(self.compute_resources[0])))

        if compute_connection_resource_name is None:
            compute_connection_resource_name = list(
                range(len(self.compute_connection_resource[0][0]))
            )

        print("\nFloorplan: device (task version)", end="")
        for t in task_nodes:
            for c in compute_nodes:
                for v_id in range(len(self.opt_placement[c][t])):
                    if self.opt_placement[c][t][v_id].x == 1:
                        print(f"{c:3d} ({v_id:d})", end=",")
                        break
        print()

        if not verbose:
            return

        print("\nFloorplan Graph:")
        # print ip ID
        print("  |", end="")
        for t in task_nodes:
            print(f"{t:2d} |", end="")
        print("<-- task_nodes")

        # print division
        print("--|", end="")
        for _t in task_nodes:
            print("---|", end="")
        print()

        # print location
        for c in compute_nodes:
            print(f"{c:2d}", end="|")
            for t in task_nodes:
                for v_id in range(len(self.opt_placement[c][t])):
                    if self.opt_placement[c][t][v_id].x == 1:
                        print(f" {v_id:d} ", end="|")
                        break
                else:
                    print("   ", end="|")
            print()

        print("\n^\n|\ncompute_nodes")

        print("\nResource results:")
        print("   |", end="")
        for r in range(len(self.compute_resources[0])):
            print(f"{compute_resources_names[r]:8s}", end="|")
        print()

        max_per_r = [0 for r in self.compute_resources[0]]
        acc_per_r = [0 for r in self.compute_resources[0]]
        cnt_per_r = [0 for r in self.compute_resources[0]]

        for i in compute_nodes:
            print(f"{i:3d}", end="|")
            for r in range(len(self.compute_resources[0])):
                if self.compute_resources[i][r] > 0:
                    res_usage = (
                        sum(
                            [
                                sum(
                                    self.task_requirements[t][v][r] * self.opt_placement[i][t][v].x
                                    for v in task_versions[t]
                                )
                                for t in task_nodes
                            ]
                        )
                        / self.compute_resources[i][r]
                        * 100
                    )
                    print(f"{res_usage:7.2f}%", end="|")

                    # update stats
                    cnt_per_r[r] += 1
                    acc_per_r[r] += res_usage
                    if res_usage > max_per_r[r]:
                        max_per_r[r] = res_usage
                else:
                    print("{:7s}%".format(""), end="|")
            print()
        print("\nMax", end="|")
        for r in range(len(self.compute_resources[0])):
            print(f"{max_per_r[r]:7.2f}%", end="|")
        print()

        print("Avg", end="|")
        for r in range(len(self.compute_resources[0])):
            print(f"{acc_per_r[r] / cnt_per_r[r]:7.2f}%", end="|")
        print("\n*Avg: Average considering only used devices\n")

        print("Additional constrains:")
        for resource_numbers, limit in self.avg_util_constrains:
            print(
                "\n (",
                " + ".join([compute_resources_names[r] for r in resource_numbers]),
                ")/",
                len(resource_numbers),
                f"< {limit * 100:6.2f}%:",
            )
            for i in compute_nodes:
                res_usage = (
                    sum(
                        [
                            sum(
                                [
                                    sum(
                                        self.task_requirements[j][v][r]
                                        * self.opt_placement[i][j][v].x
                                        for v in task_versions[j]
                                    )
                                    for j in task_nodes
                                ]
                            )
                            / self.compute_resources[i][r]
                            for r in resource_numbers
                            if self.compute_resources[i][r] > 0
                        ]
                    )
                    / len(resource_numbers)
                    * 100
                )

                print(f"{i:3d}|{res_usage:7.2f}%|")

        print("\nConnection Matrix Stats")
        for cr in range(len(compute_connection_resource_name)):
            print("\nResource:", compute_connection_resource_name[cr])
            print("  |", end="")
            for c in compute_nodes:
                print(f"{c:7d} ", end="|")
            print()

            for o in compute_nodes:
                print(f"{o:2d}", end="|")
                for d in compute_nodes:
                    if self.compute_connection_resource[o][d][cr] > 0:
                        res_usage = (
                            sum(
                                self.opt_connection_matrix[o][d][td].x
                                * self.task_dependencies_requirements[td][cr]
                                for td in range(len(self.task_dependencies))
                            )
                            / self.compute_connection_resource[o][d][cr]
                            * 100
                        )

                        print(f"{res_usage:7.2f}%", end="|")
                    else:
                        print("{:7s} ".format(""), end="|")
                print()
            print("\n")

    def is_infeasible(
        self,
    ) -> bool:
        """Return True if no feasible or optimal solution exists."""
        return (
            self.solution_status != OptimizationStatus.OPTIMAL
            and self.solution_status != OptimizationStatus.FEASIBLE
        )

    def show_edge_stats(
        self,
    ) -> None:
        """Print edge statistics for the best solution."""
        if (
            self.solution_status != OptimizationStatus.OPTIMAL
            and self.solution_status != OptimizationStatus.FEASIBLE
        ):
            print("No solution")
            return

        print("Edge    : Cost   | Compute connection")
        compute_nodes = list(range(len(self.opt_placement)))
        for td in range(len(self.task_dependencies)):
            print(
                "{:2d} - {:2d} : {:6.0f} ".format(
                    *self.task_dependencies[td],
                    sum(
                        [
                            self.opt_connection_matrix[o][d][td].x
                            * self.compute_connection_cost[o][d]
                            for o in compute_nodes
                            for d in compute_nodes
                        ]
                    ),
                ),
                end="| ",
            )
            for o in compute_nodes:
                for d in compute_nodes:
                    if self.opt_connection_matrix[o][d][td].x == 1:
                        print(o, " -->", d)

        print()

    def get_solution(self, solution_num: int = 0) -> list[dict[str, int]] | None:
        """Return the specified solution as a list of device/version assignment dicts."""
        if (
            self.solution_status != OptimizationStatus.OPTIMAL
            and self.solution_status != OptimizationStatus.FEASIBLE
        ):
            print("No solution")
            return None

        if self.model.num_solutions - 1 < solution_num:
            print(
                "There are",
                self.model.num_solutions,
                "solutions. Solution number ",
                solution_num,
                "is not correct",
            )
            return None

        task_nodes = list(range(len(self.opt_placement[0])))
        compute_nodes = list(range(len(self.opt_placement)))

        solution = []
        for t in task_nodes:
            for c in compute_nodes:
                for v_id in range(len(self.opt_placement[c][t])):
                    if self.opt_placement[c][t][v_id].xi(solution_num) == 1:
                        solution += [{"device": c, "version": v_id}]
                        break

        return solution

    def show_all_solutions(self, compute_resources_names: list[str | int] | None = None) -> None:
        """Print resource utilization for all available solutions."""
        if self.model.num_solutions < 2:
            self.report_best_solution()
            return

        task_nodes = list(range(len(self.opt_placement[0])))
        compute_nodes = list(range(len(self.opt_placement)))
        task_versions = [list(range(len(self.task_requirements[task]))) for task in task_nodes]

        if compute_resources_names is None:
            compute_resources_names = list(range(len(self.compute_resources[0])))

        for k in range(self.model.num_solutions):
            print(f"Solution {k}: Cost({self.model.objective_values[k]})")

            print("\nFloorplan: device (task version)", end="")
            for t in task_nodes:
                for c in compute_nodes:
                    for v_id in range(len(self.opt_placement[c][t])):
                        if self.opt_placement[c][t][v_id].xi(k) == 1:
                            print(f"{c:3d} ({v_id:d})", end=",")
                            break
            print()

            print("\nFloorplan Graph:")
            # print ip ID
            print("  |", end="")
            for t in task_nodes:
                print(f"{t:2d} |", end="")
            print("<-- task_nodes")

            # print division
            print("--|", end="")
            for _t in task_nodes:
                print("---|", end="")
            print()

            # print location
            for c in compute_nodes:
                print(f"{c:2d}", end="|")
                for t in task_nodes:
                    for v_id in range(len(self.opt_placement[c][t])):
                        if self.opt_placement[c][t][v_id].xi(k) == 1:
                            print(f" {v_id:d} ", end="|")
                            break
                    else:
                        print("   ", end="|")
                print()

            print("\n^\n|\ncompute_nodes")

            print("\nResource results:")
            print("   |", end="")
            for r in range(len(self.compute_resources[0])):
                print(f"{compute_resources_names[r]:8s}", end="|")
            print()

            max_per_r = [0 for r in self.compute_resources[0]]
            acc_per_r = [0 for r in self.compute_resources[0]]
            cnt_per_r = [0 for r in self.compute_resources[0]]

            for i in compute_nodes:
                print(f"{i:3d}", end="|")
                for r in range(len(self.compute_resources[0])):
                    if self.compute_resources[i][r] > 0:
                        res_usage = (
                            sum(
                                [
                                    sum(
                                        self.task_requirements[t][v][r]
                                        * self.opt_placement[i][t][v].xi(k)
                                        for v in task_versions[t]
                                    )
                                    for t in task_nodes
                                ]
                            )
                            / self.compute_resources[i][r]
                            * 100
                        )
                        print(f"{res_usage:7.2f}%", end="|")

                        # update stats
                        cnt_per_r[r] += 1
                        acc_per_r[r] += res_usage
                        if res_usage > max_per_r[r]:
                            max_per_r[r] = res_usage
                    else:
                        print("{:7s}%".format(""), end="|")
                print()
            print("\nMax", end="|")
            for r in range(len(self.compute_resources[0])):
                print(f"{max_per_r[r]:7.2f}%", end="|")
            print()

            print("Avg", end="|")
            for r in range(len(self.compute_resources[0])):
                print(f"{acc_per_r[r] / cnt_per_r[r]:7.2f}%", end="|")
            print("\n*Avg: Average considering only used devices\n")

            for _t in task_nodes:
                print("#####", end="")
            print("\n")

    def draw_tasks_graph(
        self,
        with_labels: bool = True,
        node_size: int = 1,
        alpha: float = 0.3,
        arrows: bool = True,
    ) -> None:
        """Draw the task dependency graph using networkx circular layout."""
        dg = nx.DiGraph()
        # plt.figure()
        dg.add_edges_from(self.task_dependencies)
        # Make the graph
        nx.draw_circular(
            dg, with_labels=with_labels, node_size=node_size, alpha=alpha, arrows=arrows
        )


def replicate_net(
    vertices: list[list[tuple[int, ...]]],
    edges: list[tuple[int, int]],
    edge_costs: list[tuple[int, int]],
    abs_anchors: list[tuple[int, list[int]]] | None = None,
    rel_anchors: list[tuple[int, int]] | None = None,
    slr_per_device: int = 1,
    devices: int = 1,
    replicas: int = 1,
) -> tuple[
    list[list[tuple[int, ...]]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, list[int]]],
    list[tuple[int, int]],
]:
    """Duplicate a net, creating a graph with identical disjoint sub-graphs."""
    if abs_anchors is None:
        abs_anchors = []
    if rel_anchors is None:
        rel_anchors = []
    ret_vertices = []
    ret_edge_costs = []
    ret_edges = []
    ret_abs_anchors = []
    ret_rel_anchors = []
    for i in range(replicas):
        ret_vertices += vertices
        ret_edge_costs += edge_costs
        # construct edges for the multi-net graph
        for edge in edges:
            ret_edges.append((edge[0] + i * len(vertices), edge[1] + i * len(vertices)))
        # for each of the replicas apply same abs/rel anchors
        for anchor in abs_anchors:
            anchor_node = (anchor[0] + len(vertices)) % len(vertices)
            anchor_slr = []
            for d in range(devices):
                anchor_slr += [slr + d * slr_per_device for slr in anchor[1]]
            ret_abs_anchors += [(anchor_node + i * len(vertices), anchor_slr)]
        for anchor in rel_anchors:
            anchor_node1 = (anchor[0] + len(vertices)) % len(vertices)
            anchor_node2 = (anchor[1] + len(vertices)) % len(vertices)
            ret_rel_anchors += [
                (anchor_node1 + i * len(vertices), anchor_node2 + i * len(vertices))
            ]
    return ret_vertices, ret_edges, ret_edge_costs, ret_abs_anchors, ret_rel_anchors


def res_estimation_complete(
    model: ModelWrapper, fpgapart: str, multivariant: bool = True
) -> dict[str, list[dict[str, Any]]]:
    """Estimates the resources needed for the given model and all values for
    resource-related switches.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : [{config: {}, estimate: resource estimation(s)}]}.
    """
    res_dict = {}
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            op_type = node.op_type
            inst = cast("HWCustomOp", getCustomOp(node))
            if multivariant:
                if op_type.startswith(("MVAU", "VVAU")):
                    orig_restype = inst.get_nodeattr("resType")
                    res_dict[node.name] = []
                    for restype in ["dsp", "lut"]:
                        inst.set_nodeattr("resType", restype)
                        config = {"resType": restype}
                        res_dict[node.name].append(
                            {"config": config, "estimate": inst.node_res_estimation(fpgapart)}
                        )
                    inst.set_nodeattr("resType", orig_restype)
                elif op_type.startswith("ConvolutionInputGenerator"):
                    orig_ramstyle = inst.get_nodeattr("ram_style")
                    res_dict[node.name] = []
                    for restype in ["block", "distributed", "ultra"]:
                        inst.set_nodeattr("ram_style", restype)
                        config = {"ram_style": restype}
                        res_dict[node.name].append(
                            {"config": config, "estimate": inst.node_res_estimation(fpgapart)}
                        )
                    inst.set_nodeattr("ram_style", orig_ramstyle)
                elif op_type == "StreamingFIFO_rtl":
                    orig_ramstyle = inst.get_nodeattr("ram_style")
                    orig_impl_style = inst.get_nodeattr("impl_style")
                    res_dict[node.name] = []
                    inst.set_nodeattr("impl_style", "vivado")
                    for restype in ["block", "distributed", "ultra"]:
                        inst.set_nodeattr("ram_style", restype)
                        config = {"impl_style": "vivado", "ram_style": restype}
                        res_dict[node.name].append(
                            {"config": config, "estimate": inst.node_res_estimation(fpgapart)}
                        )
                    inst.set_nodeattr("ram_style", orig_ramstyle)
                    inst.set_nodeattr("impl_style", orig_impl_style)
                else:
                    res_dict[node.name] = [
                        {"config": {}, "estimate": inst.node_res_estimation(fpgapart)}
                    ]
            else:
                res_dict[node.name] = [
                    {"config": {}, "estimate": inst.node_res_estimation(fpgapart)}
                ]

    return res_dict


#######################
# Additional constrains
#######################

# absolute anchors (restrict a task (any version) to be placed in a list of devices):
# list of 2-tuples (task, compute_node_list)
# example: 1. anchor task 0 to device 0
#          2. limit last task to device 0 and 4
# abs_anchors = [(0,[0]),(-1,[0,4])]
# abs_anchors = [(0,[0])]

# relative anchors (force task (any version) to be in the same device that other task):
# list of 2-tuples (task 1,task 2)
# example: task 0 and task 1 have to be in the same device
# rel_anchors= [(0,1),]
# rel_anchors = [(0,-1)]


def partition(
    model: ModelWrapper,
    target_clk_ns: float,
    target_platform: str = "U250",
    ndevices: int = 1,
    nreplicas: int = 1,
    abs_anchors: list[tuple[int, list[int]]] | None = None,
    rel_anchors: list[tuple[int, int]] | None = None,
    timeout: int = 300,
    multivariant: bool = True,
    linear_cuts: bool = False,
    limits: np.ndarray = DEFAULT_RES_LIMITS,
    avg_constraints: list[tuple[tuple[int, ...], float]] = DEFAULT_AVG_CONSTRAINTS,
) -> list[dict[str, dict[str, Any]]] | None:
    """Partition a FINN model across multiple compute nodes using ILP optimization.

    Returns a list of floorplan dicts (one per replica) or None if infeasible.
    """
    if abs_anchors is None:
        abs_anchors = []
    if rel_anchors is None:
        rel_anchors = []
    # get platform
    fp_pfm = platforms[target_platform](ndevices, limits=limits, avg_constraints=avg_constraints)
    part = part_map[target_platform]
    # get resources
    resources = res_estimation_complete(model, part, multivariant=multivariant)
    # post-process into list of lists
    task_requirements = []
    for key in resources:
        current_task_requirements = []
        for i in range(len(resources[key])):
            luts = resources[key][i]["estimate"]["LUT"]
            brams = resources[key][i]["estimate"]["BRAM_18K"]
            urams = resources[key][i]["estimate"]["URAM"]
            dsps = resources[key][i]["estimate"]["DSP"]
            current_task_requirements.append((luts, 0, brams, urams, dsps, 0))
        task_requirements.append(current_task_requirements)

    # get connectivity
    graph_edges = []
    node_list = list(model.graph.node)  # I also need the list to remove the nodes
    edge_costs = []
    nbranches = 0
    max_bps = int(10**9 * fp_pfm.eth_gbps)
    max_sll = max(max(fp_pfm.sll_count))
    for node_idx, n in enumerate(node_list):
        node_pred = model.find_direct_predecessors(n)
        if node_pred is None:
            # Will also eliminate nodes that are floating around for some reason
            continue
        # save edge information
        node_dependencies = [node_list.index(pred) for pred in node_pred]
        for dep in node_dependencies:
            # connectivity information
            graph_edges.append((dep, node_idx))
            # edge weights
            inst = cast("HWCustomOp", getCustomOp(model.graph.node[dep]))
            nwires = max_sll + 1 if nbranches > 0 else inst.get_outstream_width_padded()
            if nbranches > 0 or inst.get_exp_cycles() == 0:
                nbps = max_bps
            else:
                nbps = int(
                    10**9
                    * (inst.get_outstream_width_padded() * inst.get_number_output_values())
                    / (target_clk_ns * inst.get_exp_cycles())
                )
            edge_costs.append((nwires, nbps))
        # keep track of how many branches the model has at any point
        # useful if we want to avoid cutting in nonlinear segments
        if linear_cuts:
            if n.op_type.startswith("DuplicateStreams"):
                nbranches += 1
            elif n.op_type.startswith("AddStreams"):
                nbranches -= 1
    # replicate the net as required
    (
        task_requirements,
        graph_edges,
        edge_costs,
        abs_anchors,
        rel_anchors,
    ) = replicate_net(
        task_requirements,
        graph_edges,
        edge_costs,
        abs_anchors,
        rel_anchors,
        slr_per_device=fp_pfm.nslr,
        devices=fp_pfm.ndevices,
        replicas=nreplicas,
    )

    partitioner = ILP_partitioner()
    partitioner.create_model(
        task_requirements,
        graph_edges,
        edge_costs,
        fp_pfm.guide_resources,
        fp_pfm.compute_connection_cost,
        fp_pfm.compute_connection_resource,
        fp_pfm.res_limits,
        abs_anchors,
        rel_anchors,
    )
    for avg_resources, avg_limit in fp_pfm.avg_constraints:
        partitioner.add_average_of_utilizations_constrain(avg_resources, avg_limit)
    partitioner.solve_model(max_seconds=timeout)
    # if problem is infeasible, return None
    if partitioner.is_infeasible():
        return None

    partitioner.report_best_solution(
        ["LUT", "FF", "BRAMs", "URAM", "DSPs"], ["SLL", "Eth Mbps"], verbose=True
    )
    solution = partitioner.get_solution()
    if solution is None:
        return None

    initial_floorplan = model.analysis(floorplan_params)
    floorplans = []

    i = 0
    for _replica in range(nreplicas):
        floorplan = initial_floorplan
        for key in floorplan:
            if key == "Defaults":
                continue
            (
                floorplan[key]["slr"],
                floorplan[key]["device_id"],
            ) = fp_pfm.map_device_to_slr(solution[i]["device"])
            version = solution[i]["version"]
            config = resources[key][version]["config"]
            for attr in config:
                floorplan[key][attr] = config[attr]
            i += 1
        floorplans += [copy.deepcopy(floorplan)]
    return floorplans
