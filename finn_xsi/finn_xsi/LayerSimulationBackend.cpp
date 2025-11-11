#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <boost/program_options/options_description.hpp>
#include <iostream>
#include <boost/program_options.hpp>
#include <chrono>
#include <format>

#define NDEBUG
#include <rtlsim_config.hpp>
#include <Simulation.hpp>

namespace po = boost::program_options;

constexpr bool CommunicateWithPredecessor = (RTLSimConfig::NodeIndex != 0);
constexpr bool CommunicateWithSuccessor = (RTLSimConfig::NodeIndex != RTLSimConfig::TotalNodes - 1);
constexpr std::size_t InstreamCount = RTLSimConfig::istream_descs.size();
constexpr std::size_t OutstreamCount = RTLSimConfig::ostream_descs.size();

int main(int argc, const char* argv[]) {
    // Parse CLI options
    po::options_description desc{"Options"};
    desc.add_options()
        ("output,o", po::value<std::string>()->default_value("simulation_data.json"), "Simulation Data Output");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Construct simulation
    SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex,
        RTLSimConfig::TotalNodes, CommunicateWithPredecessor, CommunicateWithSuccessor> sim(
        RTLSimConfig::kernel_libname,
        RTLSimConfig::design_libname,
        "xsim_log_file.txt",
        "trace_file.txt",
        RTLSimConfig::istream_descs,
        RTLSimConfig::ostream_descs,
        RTLSimConfig::previousNodeName,
        RTLSimConfig::currentNodeName,
        2,
        vm["output"].as<std::string>()
    );
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sim.start();
    return 0;
}
