# C simulation testbenches of the LSP receiver components (built via ../CMakeLists.txt).
# One testbench per component file. A file with several components gets one test per component,
# selected by the first argument. Exit code 77 marks a test that is not implemented yet.
function(receiver_tb file)
    add_executable(${file}_tb ${file}_tb.cpp)
    foreach(test IN LISTS ARGN)
        add_test(NAME ${file}.${test} COMMAND ${file}_tb ${test})
        set_tests_properties(${file}.${test} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
endfunction()

receiver_tb(utils ctrl_status mac_dispatcher cmac_ingress cmac_egress)
receiver_tb(manager manager)
receiver_tb(connection connection)
receiver_tb(controller_tx controller_tx)
