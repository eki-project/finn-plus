# C simulation testbenches of the LSP kernels, built with the host compiler.
#   cmake -S . -B build && cmake --build build && ctest --test-dir build [-R <test>]
cmake_minimum_required(VERSION 3.20)
project(lsp_kernels LANGUAGES CXX)

# C++14 as in Vitis HLS 2024.2
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(HLS_INCLUDE_DIR ap_int.h HINTS "$ENV{XILINX_HLS}/include" NO_DEFAULT_PATH REQUIRED)
include_directories(common)
include_directories(SYSTEM "${HLS_INCLUDE_DIR}")

# Configurable constants of common/constants.hpp. Each is a cache variable named like its macro:
# left empty, the default from constants.hpp applies; set, it overrides it, e.g.
#   cmake -S . -B build -DLSP_ETHERNET_MTU=9000
# Reset to the default with -DLSP_ETHERNET_MTU=
# A -D on the cmake command line only sets a CMake variable, so the loop passes each set one on to
# the compiler as a macro definition.
set(LSP_CONFIGURABLE_CONSTANTS
    LSP_ETHERNET_MTU
)
foreach(name IN LISTS LSP_CONFIGURABLE_CONSTANTS)
    set(${name} "" CACHE STRING "Override of ${name} (empty: default from constants.hpp)")
    # Value of the variable whose name is stored in `name`
    set(value "${${name}}")
    if(NOT "${value}" STREQUAL "")
        add_compile_definitions(${name}=${value})
        message(STATUS "${name} = ${value}")
    endif()
endforeach()

enable_testing()

add_subdirectory(receiver)
