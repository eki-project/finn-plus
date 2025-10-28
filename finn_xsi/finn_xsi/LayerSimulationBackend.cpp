#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <iostream>
#include <chrono>
#include <rtlsim_config.hpp>

#define NDEBUG
#include <Simulation.hpp>

int main(){
    // TODO: Give proper names for previous and name
    constexpr bool communicateWithPredecessor = (NodeIndex != 0);
    constexpr bool communicateWithSuccessor = (NodeIndex != TotalNodes - 1);
    SingleNodeSimulation<1, 1, false, NodeIndex, TotalNodes, communicateWithPredecessor, communicateWithSuccessor> sim(
        kernel_libname,
        design_libname,
        "xsim_log_file.txt",
        "trace_file.txt",
        std::array<StreamDescriptor, 1>{StreamDescriptor{istream_descs[0].name, istream_descs[0].job_size, istream_descs[0].job_ticks}},
        std::array<StreamDescriptor, 1>{StreamDescriptor{ostream_descs[0].name, ostream_descs[0].job_size, ostream_descs[0].job_ticks}},
        previousNodeName,
        currentNodeName
    );

    // TODO: Run correct frames

    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t j = 0; j < 100000; ++j) {
        sim.runSingleCycle();
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    if constexpr(NodeIndex == 0) {
        std::cout << duration << " ms" << std::endl;
    }
    return 0;
}
