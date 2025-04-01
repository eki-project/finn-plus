#!/bin/bash 
cd /home/rz/project/finn/build_dir/vivado_stitch_proj_i3j2gd2v
vivado -mode batch -source make_project.tcl
cd /home/rz/project/finn-examples/build/vgg10-radioml
