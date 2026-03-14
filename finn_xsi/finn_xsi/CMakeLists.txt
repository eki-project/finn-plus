cmake_minimum_required(VERSION 3.11)
project(LayerSimulationBackend)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Require C++20
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Using C++ Standard ${CMAKE_CXX_STANDARD}")
SET(CMAKE_COLOR_MAKEFILE ON)

message(STATUS "CMake cwd: ${CMAKE_CURRENT_SOURCE_DIR}")

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# INCLUDES

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
    INTERFACE -Ofast -ffast-math -march=native -mtune=native -fstack-protector-strong -fopenmp -ffunction-sections -fdata-sections -pipe -funroll-loops -shared -fPIC -Wno-interference-size
    # Additional performance options:
    -flto=auto                    # Link-time optimization (auto-detect thread count)
    -fno-plt                      # Avoid PLT for better performance with shared libs
    -fno-semantic-interposition   # Allow more aggressive optimization in shared libs
    -ftree-vectorize              # Enable auto-vectorization (usually on with -O3)
    -fvect-cost-model=dynamic     # Better vectorization cost model
    -fprefetch-loop-arrays        # Prefetch arrays in loops
    -fno-math-errno               # Don't set errno for math functions (covered by -ffast-math mostly)
    -fno-trapping-math            # Allow optimizations that may trap (part of -ffast-math)
    -ffinite-math-only            # Assume no NaN/Inf (part of -ffast-math)
    -fassociative-math            # Allow reassociation (part of -ffast-math)
    )
    target_link_options(
    fifosim_options
    INTERFACE
    -flto=auto                    # LTO at link time
    -Wl,--gc-sections             # Remove unused sections
    -Wl,--as-needed               # Only link needed libraries
    -Wl,-O3                       # Linker optimization level
    -Wl,--hash-style=gnu          # Faster symbol lookup
)
  #target_link_options(fifosim_options INTERFACE -fsanitize=undefined,address)
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

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache for builds")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

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

# Collect source files
file(GLOB_RECURSE CORE_SRC src/*.cpp)

# For JSON writing
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

# Add boost for PO
find_package(Boost COMPONENTS program_options REQUIRED)

# Build the simulation library
add_library(SimulationBackendLib SHARED ${CORE_SRC})
target_include_directories(SimulationBackendLib PUBLIC "${CMAKE_BINARY_DIR}") # Include the rtlsim wrapper directory itself
target_include_directories(SimulationBackendLib PUBLIC "$ENV{XILINX_VIVADO}/data/xsim/include") # Add xsim includes
target_include_directories(SimulationBackendLib PUBLIC "include")
target_link_libraries(SimulationBackendLib PUBLIC fifosim::options nlohmann_json::nlohmann_json Threads::Threads OpenMP::OpenMP_CXX -ldl -lrt)

# Build the executable for connected simulations
add_executable(LayerSimulationBackend LayerSimulationBackend.cpp)
target_include_directories(LayerSimulationBackend SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(LayerSimulationBackend SimulationBackendLib Boost::program_options)

# Build the executable for isolated simulations
add_executable(IsolatedSimulationBackend IsolatedSimulationBackend.cpp)
target_include_directories(IsolatedSimulationBackend SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(IsolatedSimulationBackend SimulationBackendLib Boost::program_options)

OPTION(ENABLE_UNITTESTS "Enable unittests" OFF)
if(${ENABLE_UNITTESTS})
add_subdirectory(unittests)
endif()
