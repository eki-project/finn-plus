# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "MAX_FANOUT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "STAGES" -parent ${Page_0}


}

proc update_PARAM_VALUE.MAX_FANOUT { PARAM_VALUE.MAX_FANOUT } {
	# Procedure called to update MAX_FANOUT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.MAX_FANOUT { PARAM_VALUE.MAX_FANOUT } {
	# Procedure called to validate MAX_FANOUT
	return true
}

proc update_PARAM_VALUE.STAGES { PARAM_VALUE.STAGES } {
	# Procedure called to update STAGES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.STAGES { PARAM_VALUE.STAGES } {
	# Procedure called to validate STAGES
	return true
}


proc update_MODELPARAM_VALUE.STAGES { MODELPARAM_VALUE.STAGES PARAM_VALUE.STAGES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.STAGES}] ${MODELPARAM_VALUE.STAGES}
}

proc update_MODELPARAM_VALUE.MAX_FANOUT { MODELPARAM_VALUE.MAX_FANOUT PARAM_VALUE.MAX_FANOUT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.MAX_FANOUT}] ${MODELPARAM_VALUE.MAX_FANOUT}
}
