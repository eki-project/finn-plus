# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "AXI_IBITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "AXI_OBITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IBITS" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OBITS" -parent ${Page_0}


}

proc update_PARAM_VALUE.AXI_IBITS { PARAM_VALUE.AXI_IBITS } {
	# Procedure called to update AXI_IBITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_IBITS { PARAM_VALUE.AXI_IBITS } {
	# Procedure called to validate AXI_IBITS
	return true
}

proc update_PARAM_VALUE.AXI_OBITS { PARAM_VALUE.AXI_OBITS } {
	# Procedure called to update AXI_OBITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.AXI_OBITS { PARAM_VALUE.AXI_OBITS } {
	# Procedure called to validate AXI_OBITS
	return true
}

proc update_PARAM_VALUE.IBITS { PARAM_VALUE.IBITS } {
	# Procedure called to update IBITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IBITS { PARAM_VALUE.IBITS } {
	# Procedure called to validate IBITS
	return true
}

proc update_PARAM_VALUE.OBITS { PARAM_VALUE.OBITS } {
	# Procedure called to update OBITS when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OBITS { PARAM_VALUE.OBITS } {
	# Procedure called to validate OBITS
	return true
}


proc update_MODELPARAM_VALUE.IBITS { MODELPARAM_VALUE.IBITS PARAM_VALUE.IBITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IBITS}] ${MODELPARAM_VALUE.IBITS}
}

proc update_MODELPARAM_VALUE.OBITS { MODELPARAM_VALUE.OBITS PARAM_VALUE.OBITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OBITS}] ${MODELPARAM_VALUE.OBITS}
}

proc update_MODELPARAM_VALUE.AXI_IBITS { MODELPARAM_VALUE.AXI_IBITS PARAM_VALUE.AXI_IBITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_IBITS}] ${MODELPARAM_VALUE.AXI_IBITS}
}

proc update_MODELPARAM_VALUE.AXI_OBITS { MODELPARAM_VALUE.AXI_OBITS PARAM_VALUE.AXI_OBITS } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.AXI_OBITS}] ${MODELPARAM_VALUE.AXI_OBITS}
}

