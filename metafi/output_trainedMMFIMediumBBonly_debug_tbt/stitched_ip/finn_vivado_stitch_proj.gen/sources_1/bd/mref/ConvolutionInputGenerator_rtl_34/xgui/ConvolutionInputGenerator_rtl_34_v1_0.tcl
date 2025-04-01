# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BIT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BUF_IN_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BUF_OUT_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IN_WIDTH_PADDED" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MMV_IN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "MMV_OUT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OUT_WIDTH_PADDED" -parent ${Page_0}
  ipgui::add_param $IPINST -name "SIMD" -parent ${Page_0}


}

proc update_PARAM_VALUE.BIT_WIDTH { PARAM_VALUE.BIT_WIDTH } {
	# Procedure called to update BIT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BIT_WIDTH { PARAM_VALUE.BIT_WIDTH } {
	# Procedure called to validate BIT_WIDTH
	return true
}

proc update_PARAM_VALUE.BUF_IN_WIDTH { PARAM_VALUE.BUF_IN_WIDTH } {
	# Procedure called to update BUF_IN_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BUF_IN_WIDTH { PARAM_VALUE.BUF_IN_WIDTH } {
	# Procedure called to validate BUF_IN_WIDTH
	return true
}

proc update_PARAM_VALUE.BUF_OUT_WIDTH { PARAM_VALUE.BUF_OUT_WIDTH } {
	# Procedure called to update BUF_OUT_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BUF_OUT_WIDTH { PARAM_VALUE.BUF_OUT_WIDTH } {
	# Procedure called to validate BUF_OUT_WIDTH
	return true
}

proc update_PARAM_VALUE.IN_WIDTH_PADDED { PARAM_VALUE.IN_WIDTH_PADDED } {
	# Procedure called to update IN_WIDTH_PADDED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IN_WIDTH_PADDED { PARAM_VALUE.IN_WIDTH_PADDED } {
	# Procedure called to validate IN_WIDTH_PADDED
	return true
}

proc update_PARAM_VALUE.MMV_IN { PARAM_VALUE.MMV_IN } {
	# Procedure called to update MMV_IN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MMV_IN { PARAM_VALUE.MMV_IN } {
	# Procedure called to validate MMV_IN
	return true
}

proc update_PARAM_VALUE.MMV_OUT { PARAM_VALUE.MMV_OUT } {
	# Procedure called to update MMV_OUT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MMV_OUT { PARAM_VALUE.MMV_OUT } {
	# Procedure called to validate MMV_OUT
	return true
}

proc update_PARAM_VALUE.OUT_WIDTH_PADDED { PARAM_VALUE.OUT_WIDTH_PADDED } {
	# Procedure called to update OUT_WIDTH_PADDED when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OUT_WIDTH_PADDED { PARAM_VALUE.OUT_WIDTH_PADDED } {
	# Procedure called to validate OUT_WIDTH_PADDED
	return true
}

proc update_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to update SIMD when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.SIMD { PARAM_VALUE.SIMD } {
	# Procedure called to validate SIMD
	return true
}


proc update_MODELPARAM_VALUE.BIT_WIDTH { MODELPARAM_VALUE.BIT_WIDTH PARAM_VALUE.BIT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BIT_WIDTH}] ${MODELPARAM_VALUE.BIT_WIDTH}
}

proc update_MODELPARAM_VALUE.SIMD { MODELPARAM_VALUE.SIMD PARAM_VALUE.SIMD } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.SIMD}] ${MODELPARAM_VALUE.SIMD}
}

proc update_MODELPARAM_VALUE.MMV_IN { MODELPARAM_VALUE.MMV_IN PARAM_VALUE.MMV_IN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MMV_IN}] ${MODELPARAM_VALUE.MMV_IN}
}

proc update_MODELPARAM_VALUE.MMV_OUT { MODELPARAM_VALUE.MMV_OUT PARAM_VALUE.MMV_OUT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MMV_OUT}] ${MODELPARAM_VALUE.MMV_OUT}
}

proc update_MODELPARAM_VALUE.IN_WIDTH_PADDED { MODELPARAM_VALUE.IN_WIDTH_PADDED PARAM_VALUE.IN_WIDTH_PADDED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IN_WIDTH_PADDED}] ${MODELPARAM_VALUE.IN_WIDTH_PADDED}
}

proc update_MODELPARAM_VALUE.OUT_WIDTH_PADDED { MODELPARAM_VALUE.OUT_WIDTH_PADDED PARAM_VALUE.OUT_WIDTH_PADDED } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OUT_WIDTH_PADDED}] ${MODELPARAM_VALUE.OUT_WIDTH_PADDED}
}

proc update_MODELPARAM_VALUE.BUF_IN_WIDTH { MODELPARAM_VALUE.BUF_IN_WIDTH PARAM_VALUE.BUF_IN_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BUF_IN_WIDTH}] ${MODELPARAM_VALUE.BUF_IN_WIDTH}
}

proc update_MODELPARAM_VALUE.BUF_OUT_WIDTH { MODELPARAM_VALUE.BUF_OUT_WIDTH PARAM_VALUE.BUF_OUT_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BUF_OUT_WIDTH}] ${MODELPARAM_VALUE.BUF_OUT_WIDTH}
}

