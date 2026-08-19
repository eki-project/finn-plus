#!/bin/sh

# --- Assumptions ---
#	Clock:		clock = 2ns
#	File:		*.vhd, *.v, *.sv all in same directory..
#	Results:	stored in ./results_$1
#	Board:		PYNQ Z-1

# test if we're passing argument to script..
if [ $# -eq 0 ]; then
	echo "Usage: vivadocompile.sh <top-level-entity-name> <fp-tcl-scripts (optional)> <clk-name (optional)> <fpga-part (optional)> <clk-period-ns (optional)> <gen-postsynth-verilog (optional)>";
	echo "<top-level-entity-name> should not contain the .v or .vhd extension";
	echo "<fp-tcl-scripts> should be a string of TCL scripts separated by #: (e.g. 'script1.tcl#:script2.tcl')";
	exit 1;
fi

# the tcl scripts to instantiate the floating point ip come as a string
# we need to split the string and load it into an array
FP_TCL_SCRIPTS="$2"
# Remove leading and trailing quotes if present
FP_TCL_SCRIPTS=$(echo "$FP_TCL_SCRIPTS" | sed -e 's/^["\x27]//' -e 's/["\x27]$//')

# use clk as default name for clock signal if not supplied.
CLK_NAME=${3:-clk}
FPGA_PART=${4:-xc7z020clg400-1}
CLK_PERIOD=${5:-2.0}
GEN_VERILOG=${6:-0}
echo $1
echo $CLK_NAME
echo $FPGA_PART
echo $CLK_PERIOD
echo $GEN_VERILOG

# Get the directory where this script is located
OLD_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd $OLD_DIR

# clean results..
rm -rf results_$1
mkdir results_$1
cd results_$1

# put all files in a prf file..
for ext in vhd v h xdc vh sv; do
    [ -n "$(ls ../*.$ext 2>/dev/null)" ] && cp ../*.$ext .
done

#put FPGA part to be used into the project compile tcl script
echo "set fpga_part \"$FPGA_PART\"" > $1.tcl
cat "$SCRIPT_DIR/vivadocompile.tcl" >> $1.tcl
if [ $GEN_VERILOG = 1 ]; then
	echo "write_verilog -mode funcsim ${1}_post_synth.v" >> $1.tcl
fi
echo "exit" >> $1.tcl

echo "set files [list \\" > sources.tcl

# Process each file type with find
for ext in h v sv vhd; do
    find . -maxdepth 1 -name "*.${ext}" -type f | while read -r file; do
        basename_file=$(basename "$file")
        echo "\"[file normalize \"\$origin_dir/$basename_file\"]\"\\" >> sources.tcl
    done
done

echo "]" >> sources.tcl
echo "add_files -norecurse -fileset \$obj \$files" >> sources.tcl

# Add floating point IP TCL scripts if provided
if [ -n "$FP_TCL_SCRIPTS" ]; then
  # Split the string by #: delimiter
  IFS='#:'
  for element in $FP_TCL_SCRIPTS; do
    if [ -n "$element" ]; then
      echo "source $element" >> sources.tcl
    fi
  done
  unset IFS
fi

# Handle headers with find
touch headers.tcl
find . -maxdepth 1 -name "*.h" -type f | while read -r file; do
    basename_file=$(basename "$file")
    echo "set file \"\$origin_dir/$basename_file\"" >> headers.tcl
    echo "set file [file normalize \$file]" >> headers.tcl
    echo "set file_obj [get_files -of_objects [get_filesets sources_1] [list \"*\$file\"]]" >> headers.tcl
    echo "set_property \"file_type\" \"Verilog Header\" \$file_obj" >> headers.tcl

done

# caution: this overwrites the local $1.xdc file if that exists
cat "$SCRIPT_DIR/vivadocompile.xdc" >> $1.xdc
sed -i "s/clk/$CLK_NAME/g" $1.xdc
sed -i "s/CLK_PERIOD_NS/$CLK_PERIOD/g" $1.xdc

for i in *.xdc
do
    if [ "$i" = "$1.xdc" ]; then
        continue
    fi
	cat $i >> $1.xdc
done
