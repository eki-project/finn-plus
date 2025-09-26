cmake_minimum_required(VERSION 3.10)
project(FIFO_Simulation)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Additional flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

# Check if OpenMPI was found
find_package(MPI)
if (MPI_FOUND)
    message("MPI found. Using parallel iterative sizing.")
else()
    message("MPI not found. Proceeding normally.")
endif()

# Clear cache to set a new starting fifo size
if(FIFO_START_SIZE)
    message(WARNING "Setting FIFO_START_SIZE to ${FIFO_START_SIZE}. Simulation will skip the initial step!")
endif()

# Write configuration header
configure_file(simulation_config.h.in simulation_config.h)

# Main
add_executable(FIFO_Simulation fifosim.cpp xsi_finn.cpp clock/clock.cpp axis_control/axis_control.cpp axi_control/s_axi_control.cpp)

# Get the rtlsim wrapper directory
if(NOT DEFINED RTLSIM_WRAPPER_DIR)
    message(FATAL_ERROR "Pass or define RTLSIM_WRAPPER_DIR pointing to the FINN generated wrapper directory!")
else()
    message(STATUS "RTLSIM Wrapper directory set to ${RTLSIM_WRAPPER_DIR}")
endif()

# Include the rtlsim wrapper directory itself
target_include_directories(FIFO_Simulation PUBLIC "${RTLSIM_WRAPPER_DIR}")

# Add xsim includes
target_include_directories(FIFO_Simulation PUBLIC "$ENV{XILINX_VIVADO}/data/xsim/include")

# Link MPI if available
target_link_libraries(FIFO_Simulation MPI::MPI_CXX)

# Link libraries
target_link_libraries(FIFO_Simulation -ldl)
target_link_libraries(FIFO_Simulation -lrt)
