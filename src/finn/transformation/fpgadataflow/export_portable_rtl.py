# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a self-contained, portable RTL project from a stitched FINN design."""

import os
import re
import shutil
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

# Matches the file argument of a $readmemh call in (System)Verilog sources
_READMEMH_PATTERN = r'\$readmemh\s*\(\s*"([^"]+)"'


class ExportPortableRTL(Transformation):
    """Export a self-contained, portable RTL project from a stitched FINN design.

    This transformation copies all Verilog source files and memory initialization
    files (.dat) to a single directory structure with relative paths, suitable for
    use in external simulators (QuestaSim, ModelSim, Verilator) or non-IPI Vivado
    projects.

    Parameters
    ----------
    export_dir : str
        Target directory for the portable RTL export
    """

    def __init__(self, export_dir: str | Path) -> None:
        """Initialize the transformation with the target export directory."""
        super().__init__()
        self.export_dir = Path(export_dir)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Copy the RTL sources and memory files of the stitched design to the export directory."""
        # Verify stitched IP exists
        vivado_stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
        if not vivado_stitch_proj:
            raise FINNUserError("No stitched IP found. Run CreateStitchedIP first.")
        vivado_stitch_proj = Path(vivado_stitch_proj)
        if not vivado_stitch_proj.is_dir():
            raise FINNUserError(f"Stitched IP directory not found: {vivado_stitch_proj}")

        # Create export directory structure
        rtl_dir = self.export_dir / "rtl"
        src_dir = rtl_dir / "src"
        rtllib_dir = rtl_dir / "rtllib"
        data_dir = self.export_dir / "data"

        for d in [src_dir, rtllib_dir, data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Track all files for file list generation
        all_rtl_files = []
        rtllib_files_copied = set()

        # 1. Copy all RTL files from all_verilog_srcs.txt
        all_verilog_srcs_file = vivado_stitch_proj / "all_verilog_srcs.txt"
        if not all_verilog_srcs_file.is_file():
            raise FINNUserError(f"all_verilog_srcs.txt not found in {vivado_stitch_proj}")

        verilog_files = all_verilog_srcs_file.read_text().strip().split()

        for src_file in verilog_files:
            if not Path(src_file).is_file():
                continue
            # Skip XCI files (Vivado-specific IP)
            if src_file.endswith(".xci"):
                continue

            basename = Path(src_file).name

            if "finn-rtllib" in src_file:
                # Copy finn-rtllib files (preserving subdirectory structure)
                if src_file not in rtllib_files_copied:
                    rtllib_rel = src_file.split("finn-rtllib/")[1]
                    dest_file = rtllib_dir / rtllib_rel
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    all_rtl_files.append(f"rtl/rtllib/{rtllib_rel}")
                    rtllib_files_copied.add(src_file)
            else:
                # All other files go to src directory
                dest_file = src_dir / basename
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                    all_rtl_files.append(f"rtl/src/{basename}")

        # 2. Collect .dat files from nodes and scan src files for additional references
        dat_mapping: dict[str, str] = {}  # old_abs_path -> new_rel_path (relative to src_dir)

        def register_dat_file(src_dat: str) -> None:
            """Copy a memory file into the data directory and record its relative path."""
            if not Path(src_dat).is_file() or src_dat in dat_mapping:
                return
            dest_dat = data_dir / Path(src_dat).name
            if not dest_dat.exists():
                shutil.copy2(src_dat, dest_dat)
            # Relative path from src_dir to data_dir
            dat_mapping[src_dat] = os.path.relpath(dest_dat, src_dir)

        # Collect .dat files from all nodes
        for node in model.graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            for src_dat in self._get_node_dat_files(getCustomOp(node)):
                register_dat_file(src_dat)

        # Scan src files for any additional $readmemh references
        for fpath in self._verilog_files(src_dir):
            for match in re.findall(_READMEMH_PATTERN, fpath.read_text()):
                if Path(match).is_absolute():
                    register_dat_file(match)

        # 3. Rewrite absolute paths in all src Verilog files
        self._rewrite_verilog_paths(src_dir, dat_mapping)

        # 4. Generate file lists
        self._generate_filelist_f(all_rtl_files)
        self._generate_sources_tcl(all_rtl_files)
        self._generate_readme()

        return (model, False)

    @staticmethod
    def _verilog_files(directory: Path) -> list[Path]:
        """Return all (System)Verilog source files below a directory."""
        return [f for f in directory.rglob("*") if f.is_file() and f.suffix in (".v", ".sv")]

    def _get_node_dat_files(self, node_inst: CustomOp) -> list[str]:
        """Get all .dat memory initialization files for a node."""
        dat_files: list[str] = []

        # Try get_all_meminit_filenames if available (some RTL nodes)
        get_meminit_filenames = getattr(node_inst, "get_all_meminit_filenames", None)
        if get_meminit_filenames is not None:
            try:
                dat_files = [str(f) for f in get_meminit_filenames(abspath=True)]
            except Exception:
                dat_files = []

        # Also scan code_gen_dir_ipgen for .dat files
        try:
            ipgen_path = node_inst.get_nodeattr("code_gen_dir_ipgen")
        except Exception:
            ipgen_path = None
        if ipgen_path and Path(str(ipgen_path)).is_dir():
            for dat_file in Path(str(ipgen_path)).rglob("*.dat"):
                if str(dat_file) not in dat_files:
                    dat_files.append(str(dat_file))

        return dat_files

    def _rewrite_verilog_paths(self, rtl_dir: Path, path_mapping: dict[str, str]) -> None:
        """Rewrite absolute $readmemh paths to relative paths in Verilog files."""
        for fpath in self._verilog_files(rtl_dir):
            content = fpath.read_text()
            modified = False

            # Replace absolute paths with relative paths
            for old_path, new_path in path_mapping.items():
                if old_path in content:
                    content = content.replace(old_path, new_path)
                    modified = True

            # Handle any remaining absolute paths by basename matching
            for match in re.findall(_READMEMH_PATTERN, content):
                if Path(match).is_absolute():
                    basename = Path(match).name
                    for old, new in path_mapping.items():
                        if Path(old).name == basename:
                            content = content.replace(match, new)
                            modified = True
                            break

            if modified:
                fpath.write_text(content)

    def _generate_filelist_f(self, rtl_files: list[str]) -> None:
        """Generate filelist.f for Verilator/QuestaSim/ModelSim."""
        with (self.export_dir / "filelist.f").open("w") as f:
            f.write("// FINN Portable RTL Export - File List\n")
            f.write("// For use with Verilator, QuestaSim, ModelSim\n")
            f.write("// Usage: verilator -f filelist.f --top-module finn_design_wrapper\n")
            f.write("//        vlog -f filelist.f\n\n")

            # Add include directories
            f.write("// Include directories\n")
            f.write("+incdir+rtl/rtllib\n")
            f.write("+incdir+rtl/src\n\n")

            # Add source files (rtllib first, then src)
            f.write("// RTL library files\n")
            for rtl_file in sorted(rtl_files):
                if "rtllib" in rtl_file:
                    f.write(f"{rtl_file}\n")

            f.write("\n// Design files\n")
            for rtl_file in sorted(rtl_files):
                if "src" in rtl_file:
                    f.write(f"{rtl_file}\n")

    def _generate_sources_tcl(self, rtl_files: list[str]) -> None:
        """Generate sources.tcl for Vivado non-IPI projects."""
        with (self.export_dir / "sources.tcl").open("w") as f:
            f.write("# FINN Portable RTL Export - Vivado Sources\n")
            f.write("# For use with non-IPI Vivado projects\n")
            f.write("# Usage: source sources.tcl\n\n")

            f.write("# Get the directory containing this script\n")
            f.write("set script_dir [file dirname [info script]]\n\n")

            f.write("# Add all RTL source files\n")
            for rtl_file in sorted(rtl_files):
                f.write(f'add_files -norecurse "${{script_dir}}/{rtl_file}"\n')

            f.write("\n# Set include directories\n")
            f.write("set_property include_dirs [list \\\n")
            f.write('    "${script_dir}/rtl/rtllib" \\\n')
            f.write('    "${script_dir}/rtl/src" \\\n')
            f.write("] [current_fileset]\n")

    def _generate_readme(self) -> None:
        """Generate README with usage instructions."""
        with (self.export_dir / "README.md").open("w") as f:
            f.write("# FINN Portable RTL Export\n\n")
            f.write("This directory contains a self-contained RTL export of a FINN design.\n\n")
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write("portable_rtl/\n")
            f.write("├── rtl/\n")
            f.write("│   ├── src/        # Design source files\n")
            f.write("│   └── rtllib/     # Shared FINN RTL library modules\n")
            f.write("├── data/           # Memory initialization files (.dat)\n")
            f.write("├── filelist.f      # File list for simulators\n")
            f.write("├── sources.tcl     # Vivado source script\n")
            f.write("└── README.md       # This file\n")
            f.write("```\n\n")
            f.write("## Usage\n\n")
            f.write("### Verilator\n")
            f.write("```bash\n")
            f.write("verilator -f filelist.f --top-module finn_design_wrapper\n")
            f.write("```\n\n")
            f.write("### QuestaSim / ModelSim\n")
            f.write("```bash\n")
            f.write("vlog -f filelist.f\n")
            f.write("vsim finn_design_wrapper\n")
            f.write("```\n\n")
            f.write("### Vivado (non-IPI)\n")
            f.write("```tcl\n")
            f.write("source sources.tcl\n")
            f.write("set_property top finn_design_wrapper [current_fileset]\n")
            f.write("```\n\n")
            f.write("## Notes\n\n")
            f.write("- All `$readmemh` paths have been converted to relative paths\n")
            f.write("- Memory initialization files (.dat) are in the data/ directory\n")
            f.write("- The design uses FINN's streaming dataflow architecture\n")
