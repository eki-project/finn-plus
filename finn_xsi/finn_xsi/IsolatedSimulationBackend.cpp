#include <IsolatedSimulation.hpp>
#include <boost/program_options.hpp>
#include <SocketServer.h>
#include <rtlsim_config.hpp>
#include <thread>

namespace po = boost::program_options;



int main(int argc, const char* argv[]) {
    // Parse CLI options
    po::options_description desc{"Options"};
    desc.add_options()("socket,s", po::value<std::string>(), "Unix domain socket path for IPC");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Create simulation
    IsolatedSimulation<RTLSimConfig::istream_descs.size(), RTLSimConfig::ostream_descs.size()> sim(
        RTLSimConfig::kernel_libname,
        RTLSimConfig::design_libname,
        "xsim_log_file.txt",
        "trace_file.txt",
        RTLSimConfig::istream_descs,
        RTLSimConfig::ostream_descs,
        "readylog.txt",
        "validlog.txt"
    );



    // Create controller
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

        // Preparing thread variable
        std::optional<std::jthread> simThread = std::nullopt;
        std::mutex simMutex;

        // Command processing loop
        while (true) {
            // Read message
            auto request = server.receive_message();
            if (!request.has_value()) {
                std::cout << "Connection closed or error occurred" << std::endl;
                break;
            }

            // Process message
            std::size_t cycles = 0;
            std::string command = (*request)["command"];
            if (command == "start") {
                std::cout << "Starting simulation" << std::endl;
                if (!simThread.has_value()) {
                    simThread = std::jthread([&sim, &simMutex, &cycles](std::stop_token stop) {
                        {
                            std::lock_guard<std::mutex> guard(simMutex);
                            sim.simulate(true);
                        }
                        std::cout << "Simulation initialized. Going into main loop." << std::endl;
                        {
                            std::lock_guard<std::mutex> guard(simMutex);
                            sim.simulate(false);
                            std::cout << "Executed first cycle." << std::endl;
                            std::cout << "Status: " << sim.getStatus() << std::endl;
                            std::cout << "Is running: " << sim.isRunning() << std::endl;
                        }
                        while (!stop.stop_requested()) {
                            std::lock_guard<std::mutex> guard(simMutex);
                            if (cycles % 1000 == 0) {
                                std::cout << cycles << "   " << sim.getStatus() << std::endl;
                            }
                            sim.simulate(false);
                            ++cycles;
                        }
                    });
                    simThread->join();
                } else {
                    std::lock_guard<std::mutex> guard(simMutex);
                    sim.resume();
                }
            } else if (command == "stop") {
                std::cout << "Stopping simulation." << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                sim.halt();
                if (simThread.has_value()) {
                    simThread->request_stop();
                }
            } else if (command == "pause") {
                std::cout << "Pausing simulation." << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                if (simThread.has_value()) {
                    simThread->request_stop();
                }
            } else if (command == "status") {
                std::cout << "Sending status update." << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                server.send_message(sim.getStatus());
            } else {
                std::cout << "Unknown command " << command << std::endl;
                std::cerr << "Unknown command " << command << std::endl;
            }

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
