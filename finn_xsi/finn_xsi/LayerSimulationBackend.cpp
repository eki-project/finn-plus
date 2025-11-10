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
        ("depth,d", po::value<unsigned int>()->required(), "FIFO Depth")
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
        vm["depth"].as<unsigned int>()
    );


    /** SECTION WIP */
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t j = 0; j < 100000; ++j) {
        sim.runSingleCycle();
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    if constexpr(RTLSimConfig::NodeIndex == 0) {
        std::cout << duration << " ms" << std::endl;
    }
    /***********/

    // Write results as JSON
    auto outputPath = std::filesystem::path(vm["output"].as<std::string>());
    sim.writeResults(outputPath);

    return 0;
}
