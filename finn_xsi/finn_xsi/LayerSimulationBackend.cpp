#include <AXIS_Control.h>
#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <SocketServer.h>

#include <atomic>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <iostream>
#include <mutex>
#include <thread>

#define NDEBUG
#include <Simulation.hpp>
#include <rtlsim_config.hpp>

namespace po = boost::program_options;

constexpr std::size_t InstreamCount = RTLSimConfig::istream_descs.size();
constexpr std::size_t OutstreamCount = RTLSimConfig::ostream_descs.size();

// Simulation state management
enum class SimulationState { IDLE, CONFIGURED, RUNNING, FINISHED, ERROR };

class SimulationController {
     private:
    SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes>& sim;
    std::atomic<SimulationState> state{SimulationState::IDLE};
    std::atomic<uint64_t> current_cycles{0};
    std::atomic<uint64_t> current_samples{0};
    std::atomic<uint64_t> target_samples{0};
    std::mutex state_mutex;
    std::string error_message;
    std::jthread sim_thread;
    std::size_t fifo_depth{2};

     public:
    explicit SimulationController(SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes>& simulation)
        : sim(simulation) {}

    void configure(std::size_t depth, uint64_t samples) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (state != SimulationState::IDLE && state != SimulationState::FINISHED) {
            throw std::runtime_error("Cannot configure while simulation is running");
        }
        fifo_depth = depth;
        target_samples = samples;
        current_cycles = 0;
        current_samples = 0;
        state = SimulationState::CONFIGURED;
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
                // Configure simulation
                sim.setMaxFIFODepth(fifo_depth);
                sim.reset();

                // Run the simulation
                sim.runToStableState(stoken);

                // Update state based on completion
                if (!stoken.stop_requested()) {
                    current_samples.store(target_samples.load());
                    state = SimulationState::FINISHED;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> error_lock(state_mutex);
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
                status["cycles"] = sim.getCyclesRun();
                status["samples"] = sim.getCompletedMaps();
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
            std::size_t fifo_depth = payload.value("fifo_depth", 2ULL);
            uint64_t samples = payload.value("samples", 1ULL);
            controller.configure(fifo_depth, samples);
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
            // Include final status with FIFO utilization
            json final_status = controller.get_status();
            if (final_status.contains("fifo_utilization")) {
                response["fifo_utilization"] = final_status["fifo_utilization"];
            }
            if (final_status.contains("cycles")) {
                response["cycles"] = final_status["cycles"];
            }
            if (final_status.contains("samples")) {
                response["samples"] = final_status["samples"];
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
    desc.add_options()("socket,s", po::value<std::string>(),
                                                                                                                              "Unix domain socket path for IPC");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Construct simulation
    SingleNodeSimulation<InstreamCount, OutstreamCount, RTLSimConfig::LoggingEnabled, RTLSimConfig::NodeIndex, RTLSimConfig::TotalNodes> sim(
        RTLSimConfig::kernel_libname, RTLSimConfig::design_libname, "xsim_log_file.txt", "trace_file.txt", RTLSimConfig::istream_descs, RTLSimConfig::ostream_descs,
        RTLSimConfig::previousNodeName, RTLSimConfig::currentNodeName, 2);

    // Create simulation controller
    SimulationController controller(sim);

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
