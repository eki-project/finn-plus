#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <Simulation.hpp>
#include <iostream>
#include <rtlsim_config.hpp>

int main(){
    // TODO: Give proper names for previous and name
    constexpr bool communicateWithPredecessor = (NodeIndex != 0);
    constexpr bool communicateWithSuccessor = (NodeIndex != TotalNodes - 1);
    SingleNodeSimulation<1, 1, true, NodeIndex, TotalNodes, communicateWithPredecessor, communicateWithSuccessor> sim(
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
    sim.runForFrames(10);
    return 0;
}
