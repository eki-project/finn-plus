#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <Simulation.hpp>

int main(){
    Simulation<1, 1> sim("kernel_lib", "design_lib", "xsim_log_file", "trace_file",
                          std::array<StreamDescriptor, 1>{StreamDescriptor{"input", 1024, 10}},
                          std::array<StreamDescriptor, 1>{StreamDescriptor{"output", 1024, 10}});
    return 0;
}
