# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BIAS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEEP_PIPELINE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEPTH_TRIGGER_BRAM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "DEPTH_TRIGGER_URAM" -parent ${Page_0}
  ipgui::add_param $IPINST -name "FPARG" -parent ${Page_0}
  ipgui::add_param $IPINST -name "N" -parent ${Page_0}
  ipgui::add_param $IPINST -name "O_BITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIGNED" -parent ${Page_0}
  ipgui::add_param $IPINST -name "THRESHOLDS_PATH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "USE_AXILITE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WI" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WT" -parent ${Page_0}


}

proc update_PARAM_VALUE.BIAS { PARAM_VALUE.BIAS } {
	# Procedure called to update BIAS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BIAS { PARAM_VALUE.BIAS } {
	# Procedure called to validate BIAS
	return true
}

proc update_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to update C when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C { PARAM_VALUE.C } {
	# Procedure called to validate C
	return true
}

proc update_PARAM_VALUE.DEEP_PIPELINE { PARAM_VALUE.DEEP_PIPELINE } {
	# Procedure called to update DEEP_PIPELINE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEEP_PIPELINE { PARAM_VALUE.DEEP_PIPELINE } {
	# Procedure called to validate DEEP_PIPELINE
	return true
}

proc update_PARAM_VALUE.DEPTH_TRIGGER_BRAM { PARAM_VALUE.DEPTH_TRIGGER_BRAM } {
	# Procedure called to update DEPTH_TRIGGER_BRAM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH_TRIGGER_BRAM { PARAM_VALUE.DEPTH_TRIGGER_BRAM } {
	# Procedure called to validate DEPTH_TRIGGER_BRAM
	return true
}

proc update_PARAM_VALUE.DEPTH_TRIGGER_URAM { PARAM_VALUE.DEPTH_TRIGGER_URAM } {
	# Procedure called to update DEPTH_TRIGGER_URAM when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.DEPTH_TRIGGER_URAM { PARAM_VALUE.DEPTH_TRIGGER_URAM } {
	# Procedure called to validate DEPTH_TRIGGER_URAM
	return true
}

proc update_PARAM_VALUE.FPARG { PARAM_VALUE.FPARG } {
	# Procedure called to update FPARG when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.FPARG { PARAM_VALUE.FPARG } {
	# Procedure called to validate FPARG
	return true
}

proc update_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to update N when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.N { PARAM_VALUE.N } {
	# Procedure called to validate N
	return true
}

proc update_PARAM_VALUE.O_BITS { PARAM_VALUE.O_BITS } {
	# Procedure called to update O_BITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.O_BITS { PARAM_VALUE.O_BITS } {
	# Procedure called to validate O_BITS
	return true
}

proc update_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to update PE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PE { PARAM_VALUE.PE } {
	# Procedure called to validate PE
	return true
}

proc update_PARAM_VALUE.SIGNED { PARAM_VALUE.SIGNED } {
	# Procedure called to update SIGNED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIGNED { PARAM_VALUE.SIGNED } {
	# Procedure called to validate SIGNED
	return true
}

proc update_PARAM_VALUE.THRESHOLDS_PATH { PARAM_VALUE.THRESHOLDS_PATH } {
	# Procedure called to update THRESHOLDS_PATH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.THRESHOLDS_PATH { PARAM_VALUE.THRESHOLDS_PATH } {
	# Procedure called to validate THRESHOLDS_PATH
	return true
}

proc update_PARAM_VALUE.USE_AXILITE { PARAM_VALUE.USE_AXILITE } {
	# Procedure called to update USE_AXILITE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.USE_AXILITE { PARAM_VALUE.USE_AXILITE } {
	# Procedure called to validate USE_AXILITE
	return true
}

proc update_PARAM_VALUE.WI { PARAM_VALUE.WI } {
	# Procedure called to update WI when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WI { PARAM_VALUE.WI } {
	# Procedure called to validate WI
	return true
}

proc update_PARAM_VALUE.WT { PARAM_VALUE.WT } {
	# Procedure called to update WT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WT { PARAM_VALUE.WT } {
	# Procedure called to validate WT
	return true
}


proc update_MODELPARAM_VALUE.N { MODELPARAM_VALUE.N PARAM_VALUE.N } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.N}] ${MODELPARAM_VALUE.N}
}

proc update_MODELPARAM_VALUE.WI { MODELPARAM_VALUE.WI PARAM_VALUE.WI } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WI}] ${MODELPARAM_VALUE.WI}
}

proc update_MODELPARAM_VALUE.WT { MODELPARAM_VALUE.WT PARAM_VALUE.WT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WT}] ${MODELPARAM_VALUE.WT}
}

proc update_MODELPARAM_VALUE.C { MODELPARAM_VALUE.C PARAM_VALUE.C } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C}] ${MODELPARAM_VALUE.C}
}

proc update_MODELPARAM_VALUE.PE { MODELPARAM_VALUE.PE PARAM_VALUE.PE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PE}] ${MODELPARAM_VALUE.PE}
}

proc update_MODELPARAM_VALUE.SIGNED { MODELPARAM_VALUE.SIGNED PARAM_VALUE.SIGNED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIGNED}] ${MODELPARAM_VALUE.SIGNED}
}

proc update_MODELPARAM_VALUE.FPARG { MODELPARAM_VALUE.FPARG PARAM_VALUE.FPARG } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.FPARG}] ${MODELPARAM_VALUE.FPARG}
}

proc update_MODELPARAM_VALUE.BIAS { MODELPARAM_VALUE.BIAS PARAM_VALUE.BIAS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BIAS}] ${MODELPARAM_VALUE.BIAS}
}

proc update_MODELPARAM_VALUE.THRESHOLDS_PATH { MODELPARAM_VALUE.THRESHOLDS_PATH PARAM_VALUE.THRESHOLDS_PATH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.THRESHOLDS_PATH}] ${MODELPARAM_VALUE.THRESHOLDS_PATH}
}

proc update_MODELPARAM_VALUE.USE_AXILITE { MODELPARAM_VALUE.USE_AXILITE PARAM_VALUE.USE_AXILITE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.USE_AXILITE}] ${MODELPARAM_VALUE.USE_AXILITE}
}

proc update_MODELPARAM_VALUE.DEPTH_TRIGGER_URAM { MODELPARAM_VALUE.DEPTH_TRIGGER_URAM PARAM_VALUE.DEPTH_TRIGGER_URAM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH_TRIGGER_URAM}] ${MODELPARAM_VALUE.DEPTH_TRIGGER_URAM}
}

proc update_MODELPARAM_VALUE.DEPTH_TRIGGER_BRAM { MODELPARAM_VALUE.DEPTH_TRIGGER_BRAM PARAM_VALUE.DEPTH_TRIGGER_BRAM } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEPTH_TRIGGER_BRAM}] ${MODELPARAM_VALUE.DEPTH_TRIGGER_BRAM}
}

proc update_MODELPARAM_VALUE.DEEP_PIPELINE { MODELPARAM_VALUE.DEEP_PIPELINE PARAM_VALUE.DEEP_PIPELINE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.DEEP_PIPELINE}] ${MODELPARAM_VALUE.DEEP_PIPELINE}
}

proc update_MODELPARAM_VALUE.O_BITS { MODELPARAM_VALUE.O_BITS PARAM_VALUE.O_BITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.O_BITS}] ${MODELPARAM_VALUE.O_BITS}
}

