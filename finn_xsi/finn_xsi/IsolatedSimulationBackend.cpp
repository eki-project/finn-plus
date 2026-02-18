#include <IsolatedSimulation.hpp>
#include <boost/program_options.hpp>
#include <SocketServer.h>
#include <chrono>
#include <rtlsim_config.hpp>
#include <thread>

namespace po = boost::program_options;


std::string getTime() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto formatted = std::put_time(std::localtime(&now), "[%T]");
    std::stringstream ss;
    ss << formatted;
    return ss.str();
}


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
        "trace_file.wdb",
        RTLSimConfig::istream_descs,
        RTLSimConfig::ostream_descs
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
        std::size_t cycles = 0;
        std::size_t statusSent = 0;
        json response;
        while (true) {
            response = json::object();
            // Read message
            std::cout << getTime() << " Awaiting message..." << std::endl;
            auto request = server.receive_message();
            if (!request.has_value()) {
                std::cout << getTime() << " Connection closed or error occurred" << std::endl;
                break;
            }

            // Process message
            std::string command = (*request)["command"];
            std::cout << getTime() << " [Received command] " << command << std::endl;
            if (command == "start") {
                std::cout << getTime() << " Starting simulation" << std::endl;
                if (!simThread.has_value()) {
                    simThread = std::jthread([&sim, &simMutex, &cycles](std::stop_token stop) {
                        {
                            std::lock_guard<std::mutex> guard(simMutex);
                            sim.simulate(true);
                        }
                        std::cout << getTime() << " Simulation initialized. Going into main loop." << std::endl;
                        while (!stop.stop_requested()) {
                            std::lock_guard<std::mutex> guard(simMutex);
                            if (cycles % 10000 == 0) {
                                std::cout << cycles << "   " << sim.getStatus() << std::endl;
                            }
                            sim.simulate(false);
                            ++cycles;
                            if (sim.isDone()) {
                                // For now do not clean up the JSON logs, as this is
                                // done by the "stop" command from the python side of things.
                                // TODO: However this should be changed when the communication is
                                // rewritten
                                sim.commitLogsToDisk(false);
                                break;
                            }
                        }
                    });
                } else {
                    std::lock_guard<std::mutex> guard(simMutex);
                    sim.resume();
                }
                response["state"] = "running";
                server.send_message(response);
            } else if (command == "stop") {
                std::cout << getTime() << " Stopping simulation." << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                std::cout << getTime() << " Final status: " << sim.getStatus() << std::endl;
                std::cout << getTime() << " Is done? " << sim.isDone() << std::endl;
                sim.halt();
                if (simThread.has_value()) {
                    simThread->request_stop();
                }
                sim.commitLogsToDisk(true);
                response["state"] = "stopped";
                server.send_message(response);
            } else if (command == "pause") {
                std::cout << getTime() << " Pausing simulation." << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                if (simThread.has_value()) {
                    simThread->request_stop();
                }
                response["state"] = "halted";
                server.send_message(response);
            } else if (command == "status") {
                std::cout << getTime() << " [Sending] Sending status update " << statusSent + 1 << std::endl;
                std::lock_guard<std::mutex> guard(simMutex);
                json status = sim.getStatus();
                server.send_message(status);
                statusSent++;
                std::cout << getTime() << " [Sending] Status " << statusSent << " update sent!" << std::endl;
            } else {
                std::cout << getTime() << " Unknown command " << command << std::endl;
                std::cerr << "Unknown command " << command << std::endl;
                response["state"] = "unknown_command";
                server.send_message(response);
            }

            // Exit if stop command received
            if ((*request)["command"] == "stop") {
                break;
            }
        }
        simThread->join();
    } else {
        throw std::runtime_error("Socket path not provided. Socket communication is required.");
    }
    return 0;
}
