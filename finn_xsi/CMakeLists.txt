cmake_minimum_required(VERSION 3.10)
project(FIFO_Simulation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Require C++20
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using C++ Standard ${CMAKE_CXX_STANDARD}")
SET(CMAKE_COLOR_MAKEFILE ON)

message(STATUS "CMake cwd: ${CMAKE_CURRENT_SOURCE_DIR}")

# INCLUDES

# OPENMPI
# Check if OpenMPI was found
find_package(MPI)
if (MPI_FOUND)
    message("MPI found. Using parallel iterative sizing.")
else()
    message("MPI not found. Proceeding normally.")
endif()

#Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#OpenMP
find_package(OpenMP REQUIRED)

#Compiler Options
add_library(fifosim_options INTERFACE)
add_library(fifosim::options ALIAS fifosim_options)

OPTION(FIFOSIM_ENABLE_ALLOPT "Enable all optimizations" ON)
if(${FIFOSIM_ENABLE_ALLOPT})
  message(STATUS "All optimizations are enabled")
  target_compile_options(
    fifosim_options
    INTERFACE -Ofast -ffast-math -march=native -mtune=native -fstack-protector-strong -fopenmp -ffunction-sections -fdata-sections -pipe -funroll-loops -shared -fPIC)
endif()

### Enable compiler warnings
option(FIFOSIM_ENABLE_WARNINGS "Enable warnings" ON)
if (FIFOSIM_ENABLE_WARNINGS)
  include(cmake/CompilerWarnings.cmake)
  fifosim_set_project_warnings(
    fifosim_options
    OFF
    ""
    ""
    ""
    "")
endif (FIFOSIM_ENABLE_WARNINGS)

#
# Create options for including cmake files from the cmake folder with a bit of output.
#
macro(check_include)
  if(NOT ${ARGC} EQUAL 3)
    message(FATAL_ERROR "Call to 'check_include' with ${ARGC} arguments instead of 3")
  endif()
  OPTION(${ARGV0} "Enable ${ARGV0}" ON)
  if (${ARGV0})
    message(STATUS "${ARGV1}: enabled")
    include(cmake/${ARGV2})
  else()
    message(STATUS "${ARGV1}: disabled")
  endif()
endmacro()

message(STATUS "Checks:")
list(APPEND CMAKE_MESSAGE_INDENT "  ") #indent +1
check_include(FIFOSIM_IPO          "InterproceduralOptimization" InterproceduralOptimization.cmake)
list(POP_BACK CMAKE_MESSAGE_INDENT)    #indent -1

# Clear cache to set a new starting fifo size
if(FIFO_START_SIZE)
    message(WARNING "Setting FIFO_START_SIZE to ${FIFO_START_SIZE}. Simulation will skip the initial step!")
endif()

# Write configuration header
configure_file(simulation_config.h.in simulation_config.h)

# Main
add_executable(FIFO_Simulation fifosim.cpp xsi_finn.cpp clock/clock.cpp axis_control/axis_control.cpp axi_control/s_axi_control.cpp)

# # Get the rtlsim wrapper directory
# if(NOT DEFINED RTLSIM_WRAPPER_DIR)
#     message(FATAL_ERROR "Pass or define RTLSIM_WRAPPER_DIR pointing to the FINN generated wrapper directory!")
# else()
#     message(STATUS "RTLSIM Wrapper directory set to ${RTLSIM_WRAPPER_DIR}")
# endif()

# Include the rtlsim wrapper directory itself
target_include_directories(FIFO_Simulation PUBLIC "${CMAKE_BINARY_DIR}")

# Add xsim includes
target_include_directories(FIFO_Simulation PUBLIC "$ENV{XILINX_VIVADO}/data/xsim/include")

# Link MPI if available
target_link_libraries(FIFO_Simulation fifosim::options MPI::MPI_CXX Threads::Threads OpenMP::OpenMP_CXX)

# Link libraries
target_link_libraries(FIFO_Simulation -ldl)
target_link_libraries(FIFO_Simulation -lrt)
