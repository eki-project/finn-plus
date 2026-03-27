#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <SocketServer.h>
#include <sys/stat.h>

#include <atomic>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <cstddef>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

#define NDEBUG
#include <Simulation.hpp>
#include <rtlsim_config.hpp>

namespace po = boost::program_options;

constexpr std::size_t InstreamCount = RTLSimConfig::istream_descs.size();
constexpr std::size_t OutstreamCount = RTLSimConfig::ostream_descs.size();

static_assert(InstreamCount == RTLSimConfig::inputInterfaceNames.size(), "Number of input streams must match number of previous nodes");
static_assert(OutstreamCount == RTLSimConfig::outputInterfaceNames.size(), "Number of output streams must match number of next nodes");

// Simulation state management
enum class SimulationState { IDLE, CONFIGURED, RUNNING, FINISHED, ERROR };

class SimulationController {
     private:
    SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes, RTLSimConfig::IsInputNode,
                         RTLSimConfig::IsOutputNode>& sim;
    std::atomic<SimulationState> state{SimulationState::IDLE};
    std::atomic<uint64_t> current_cycles{0};
    std::atomic<uint64_t> current_samples{0};
    std::mutex state_mutex;
    std::string error_message;
    std::jthread sim_thread;
    std::vector<std::size_t> fifo_depths{2};
    std::size_t max_cycles{std::numeric_limits<std::size_t>::max()};
    bool timeout_occurred{false};

     public:
    explicit SimulationController(SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes,
                                                       RTLSimConfig::IsInputNode, RTLSimConfig::IsOutputNode>& simulation)
        : sim(simulation) {}

    void configure(const std::vector<std::size_t>& depths, const std::vector<std::size_t>& expected_first_valid_cycles, std::size_t maxCycles) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (state != SimulationState::IDLE && state != SimulationState::FINISHED) {
            throw std::runtime_error("Cannot configure while simulation is running");
        }
        fifo_depths = depths;
        current_cycles = 0;
        current_samples = 0;
        max_cycles = maxCycles;
        state = SimulationState::CONFIGURED;

        // Reset simulation first
        sim.reset();

        // Configure FIFO depths AFTER reset
        std::size_t num_fifos = sim.getFIFOCount();

        if (fifo_depths.empty()) {
            throw std::runtime_error("FIFO depths not configured");
        }

        // Apply depths: if list is shorter, use last value for remaining FIFOs
        for (std::size_t i = 0; i < num_fifos; ++i) {
            std::size_t depth_idx = std::min(i, fifo_depths.size() - 1);
            sim.setFIFODepth(i, fifo_depths[depth_idx]);
        }

        for (std::size_t i = 0; i < expected_first_valid_cycles.size(); ++i) {
            std::size_t cycles_idx = std::min(i, expected_first_valid_cycles.size() - 1);
            sim.setFIFOCyclesUntilExpectedFirstValid(i, expected_first_valid_cycles[cycles_idx]);
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (state != SimulationState::CONFIGURED) {
            throw std::runtime_error("Simulation must be configured before starting");
        }

        state = SimulationState::RUNNING;

        // Start simulation in a separate thread
        sim_thread = std::jthread([this](std::stop_token stoken) {
            try {
                std::cout << "Starting simulation with max cycles: " << max_cycles << std::endl;

                // Run the simulation
                bool timeout = sim.runToStableState(stoken, max_cycles);

                if (timeout) {
                    state = SimulationState::FINISHED;
                    timeout_occurred = true;
                }

                // Update state based on completion
                if (!stoken.stop_requested()) {
                    current_samples.store(sim.getCompletedMaps());
                    state = SimulationState::FINISHED;
                }
                state = SimulationState::FINISHED;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> error_lock(state_mutex);
                std::cout << "Simulation error: " << e.what() << std::endl;
                error_message = e.what();
                state = SimulationState::ERROR;
            }
        });
    }

    void stop() {
        if (sim_thread.joinable()) {
            sim_thread.request_stop();
            sim_thread.join();
        }
        if (state == SimulationState::RUNNING) {
            state = SimulationState::FINISHED;
        }
    }

    json get_status() const {
        json status;
        status["status"] = "success";

        SimulationState current_state = state.load();
        switch (current_state) {
            case SimulationState::IDLE:
                status["state"] = "idle";
                break;
            case SimulationState::CONFIGURED:
                status["state"] = "configured";
                break;
            case SimulationState::RUNNING:
                status["state"] = "running";
                status["cycles"] = sim.getCyclesRun();
                status["samples"] = sim.getCompletedMaps();
                break;
            case SimulationState::FINISHED:
                status["state"] = "finished";
                status["timeout"] = timeout_occurred;
                if (timeout_occurred) {
                    status["state"] = "timeout";
                }
                status["cycles"] = sim.getCyclesRun();
                status["samples"] = sim.getCompletedMaps();
                status["intervals"] = sim.getOStreamStableStateIntervals();
                // Add FIFO depth data
                {
                    auto depths = sim.getFIFODepth();
                    json fifo_depth = json::array();
                    for (size_t i = 0; i < depths.size(); ++i) {
                        fifo_depth.push_back(depths[i]);
                    }
                    if (!fifo_depth.empty()) {
                        status["fifo_depth"] = fifo_depth;
                    }
                }
                // Add FIFO utilization data
                {
                    auto utilizations = sim.getFIFOUtilization();
                    json fifo_util = json::array();
                    for (size_t i = 0; i < utilizations.size(); ++i) {
                        fifo_util.push_back(utilizations[i]);
                    }
                    if (!fifo_util.empty()) {
                        status["fifo_utilization"] = fifo_util;
                    }
                }
                // Add FIFO cycles until first valid data
                {
                    auto cycles_until_valid = sim.getFIFOCyclesUntilFirstValid();
                    json fifo_cycles = json::array();
                    for (size_t i = 0; i < cycles_until_valid.size(); ++i) {
                        fifo_cycles.push_back(cycles_until_valid[i]);
                    }
                    if (!fifo_cycles.empty()) {
                        status["fifo_cycles_until_first_valid"] = fifo_cycles;
                    }
                }
                // Add input/output job sizes
                {
                    json in_job_sizes = json::array();
                    for (size_t i = 0; i < InstreamCount; ++i) {
                        in_job_sizes.push_back(sim.getInputJobSize(i));
                    }
                    status["input_job_size"] = in_job_sizes;

                    json out_job_sizes = json::array();
                    for (size_t i = 0; i < OutstreamCount; ++i) {
                        out_job_sizes.push_back(sim.getOutputJobSize(i));
                    }
                    status["output_job_size"] = out_job_sizes;
                }
                break;
            case SimulationState::ERROR:
                status["state"] = "error";
                status["message"] = error_message;
                break;
        }
        return status;
    }

    ~SimulationController() { stop(); }
};

void process_command(const json& request, json& response, SimulationController& controller) {
    const std::string command = request["command"];
    const json& payload = request["payload"];

    try {
        if (command == "configure") {
            std::vector<std::size_t> fifo_depths;

            // std::cout << "Payload: " << payload << std::endl;

            // Handle fifo_depth as either a single value or an array
            if (payload.contains("fifo_depth")) {
                const auto& depth_value = payload["fifo_depth"];
                if (depth_value.is_array()) {
                    for (const auto& val : depth_value) {
                        fifo_depths.push_back(val.get<std::size_t>());
                    }
                } else {
                    fifo_depths.push_back(depth_value.get<std::size_t>());
                }
            } else {
                fifo_depths.push_back(std::numeric_limits<std::size_t>::max());  // Default value
            }

            std::vector<std::size_t> expected_first_valid_cycles;
            if (payload.contains("fifo_first_valid_cycles")) {
                const auto& expected_cycles_value = payload["fifo_first_valid_cycles"];
                if (expected_cycles_value.is_array()) {
                    for (const auto& val : expected_cycles_value) {
                        expected_first_valid_cycles.push_back(val.get<std::size_t>());
                    }
                } else {
                    expected_first_valid_cycles.push_back(expected_cycles_value.get<std::size_t>());
                }
            }

            if (fifo_depths.empty()) {
                throw std::runtime_error("FIFO depth list cannot be empty");
            }

            std::size_t max_cycles = std::numeric_limits<size_t>::max();
            if (payload.contains("max_cycles")) {
                max_cycles = payload["max_cycles"].get<std::size_t>();
            }

            controller.configure(fifo_depths, expected_first_valid_cycles, max_cycles);
            response["status"] = "success";
            response["message"] = "Configuration successful";
        } else if (command == "start") {
            controller.start();
            response["status"] = "success";
            response["message"] = "Simulation started";
        } else if (command == "status") {
            response = controller.get_status();
        } else if (command == "stop") {
            controller.stop();
            response["status"] = "success";
            response["message"] = "Simulation stopped";
            // Include final status with FIFO utilization and depth
            json final_status = controller.get_status();
            if (final_status.contains("fifo_utilization")) {
                response["fifo_utilization"] = final_status["fifo_utilization"];
            }
            if (final_status.contains("fifo_depth")) {
                response["fifo_depth"] = final_status["fifo_depth"];
            }
            if (final_status.contains("cycles")) {
                response["cycles"] = final_status["cycles"];
            }
            if (final_status.contains("samples")) {
                response["samples"] = final_status["samples"];
            }
            if (final_status.contains("intervals")) {
                response["intervals"] = final_status["intervals"];
            }
            if (final_status.contains("timeout")) {
                response["timeout"] = final_status["timeout"];
            }
            if (final_status.contains("fifo_cycles_until_first_valid")) {
                response["fifo_cycles_until_first_valid"] = final_status["fifo_cycles_until_first_valid"];
            }
            if (final_status.contains("input_job_size")) {
                response["input_job_size"] = final_status["input_job_size"];
            }
            if (final_status.contains("output_job_size")) {
                response["output_job_size"] = final_status["output_job_size"];
            }
        } else {
            response["status"] = "error";
            response["message"] = "Unknown command: " + command;
        }
    } catch (const std::exception& e) {
        response["status"] = "error";
        response["message"] = std::string("Error: ") + e.what();
    }
}

int main(int argc, const char* argv[]) {
    // Parse CLI options
    po::options_description desc{"Options"};
    desc.add_options()("socket,s", po::value<std::string>(), "Unix domain socket path for IPC");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << "Connected Simulation Node Index: " << RTLSimConfig::NodeIndex << " / " << RTLSimConfig::TotalNodes << std::endl;

    // Check if socket communication is enabled
    if (vm.count("socket")) {
        const std::string socket_path = vm["socket"].as<std::string>();
        std::cout << "Initializing socket server at: " << socket_path << std::endl;
        std::cout.flush();

        SocketServer server(socket_path);
        if (auto error = server.initialize(); error.has_value()) {
            std::cerr << "Failed to initialize socket server: " << *error << std::endl;
            std::cerr.flush();
            return 1;
        }

        std::cout << "Socket server initialized, waiting for commands..." << std::endl;
        std::cout.flush();

        // Construct simulation
        SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes, RTLSimConfig::IsInputNode,
                             RTLSimConfig::IsOutputNode>
            sim(RTLSimConfig::kernel_libname, RTLSimConfig::design_libname, "xsim_log_file.txt", "trace_file.txt", RTLSimConfig::istream_descs, RTLSimConfig::ostream_descs,
                RTLSimConfig::inputInterfaceNames, RTLSimConfig::outputInterfaceNames, 2);

        // Create simulation controller
        SimulationController controller(sim);

        // Command processing loop
        while (true) {
            auto request = server.receive_message();
            if (!request.has_value()) {
                std::cout << "Connection closed or error occurred" << std::endl;
                break;
            }

            json response;
            process_command(*request, response, controller);
            server.send_message(response);

            // Exit if stop command received
            if ((*request)["command"] == "stop") {
                break;
            }
        }
    } else {
        throw std::runtime_error("Socket path not provided. Socket communication is required.");
    }

    return 0;
}
