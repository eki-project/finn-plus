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
enable_testing()

add_subdirectory(receiver)
