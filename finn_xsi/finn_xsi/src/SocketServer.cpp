#include <SocketServer.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <utility>

SocketServer::SocketServer(std::string_view path) : socket_path(path) {}

SocketServer::SocketServer(std::string_view host, std::uint16_t port) : mode(Mode::Tcp), bind_host(host), bind_port(port) {}

SocketServer::~SocketServer() { close_connection(); }

SocketServer::SocketServer(SocketServer&& other) noexcept
    : server_fd(std::exchange(other.server_fd, -1)), client_fd(std::exchange(other.client_fd, -1)), socket_path(std::move(other.socket_path)) {}

SocketServer& SocketServer::operator=(SocketServer&& other) noexcept {
    if (this != &other) {
        close_connection();
        server_fd = std::exchange(other.server_fd, -1);
        client_fd = std::exchange(other.client_fd, -1);
        socket_path = std::move(other.socket_path);
    }
    return *this;
}

void SocketServer::close_fd(int& fd) noexcept {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

std::optional<std::string> SocketServer::initialize() {
    if (mode == Mode::Unix) {
        // Create Unix-domain socket
        server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd < 0) {
            return "Failed to create socket: " + std::string(strerror(errno));
        }

        // Remove existing socket file
        unlink(socket_path.c_str());

        // Bind socket
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::string error = "Failed to bind socket: " + std::string(strerror(errno));
            close_fd(server_fd);
            return error;
        }
    } else {
        // Create TCP socket
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            return "Failed to create TCP socket: " + std::string(strerror(errno));
        }

        int opt = 1;
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            std::string error = "Failed to set SO_REUSEADDR: " + std::string(strerror(errno));
            close_fd(server_fd);
            return error;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(bind_port);

        if (bind_host.empty() || bind_host == "0.0.0.0" || bind_host == "*" || bind_host == "localhost") {
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
        } else {
            if (inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
                addrinfo hints{};
                hints.ai_family = AF_INET;
                hints.ai_socktype = SOCK_STREAM;
                addrinfo* result = nullptr;
                const int rc = getaddrinfo(bind_host.c_str(), nullptr, &hints, &result);
                if (rc != 0 || result == nullptr) {
                    std::string error = "Failed to resolve bind host '" + bind_host + "': " + gai_strerror(rc);
                    if (result != nullptr) {
                        freeaddrinfo(result);
                    }
                    close_fd(server_fd);
                    return error;
                }
                addr.sin_addr = reinterpret_cast<sockaddr_in*>(result->ai_addr)->sin_addr;
                freeaddrinfo(result);
            }
        }

        if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::string error = "Failed to bind TCP socket: " + std::string(strerror(errno));
            close_fd(server_fd);
            return error;
        }
    }

    // Listen
    if (listen(server_fd, 1) < 0) {
        std::string error = "Failed to listen on socket: " + std::string(strerror(errno));
        close_fd(server_fd);
        return error;
    }

    // Accept connection
    client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
        std::string error = "Failed to accept connection: " + std::string(strerror(errno));
        close_fd(server_fd);
        return error;
    }

    return std::nullopt;  // Success
}

std::optional<json> SocketServer::receive_message() {
    if (client_fd < 0) {
        std::cerr << "Socket not connected" << std::endl;
        return std::nullopt;
    }

    // Read length prefix
    uint32_t length{};
    const ssize_t bytes_read = read(client_fd, &length, sizeof(length));
    if (bytes_read != sizeof(length)) {
        if (bytes_read == 0) {
            std::cerr << "Connection closed by client" << std::endl;
        } else {
            std::cerr << "Failed to read message length: " << strerror(errno) << std::endl;
        }
        return std::nullopt;
    }

    // Read message
    std::string buffer(length, '\0');
    size_t total_read = 0;
    while (total_read < length) {
        const ssize_t n = read(client_fd, buffer.data() + total_read, length - total_read);
        if (n <= 0) {
            std::cerr << "Failed to read message data: " << strerror(errno) << std::endl;
            return std::nullopt;
        }
        total_read += static_cast<size_t>(n);
    }

    try {
        return json::parse(buffer);
    } catch (const json::exception& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return std::nullopt;
    }
}

void SocketServer::send_message(const json& message) {
    if (client_fd < 0) {
        std::cerr << "Socket not connected" << std::endl;
        return;
    }

    const std::string msg_str = message.dump();
    const uint32_t length = static_cast<uint32_t>(msg_str.size());

    // Send length prefix
    const ssize_t bytes_written = write(client_fd, &length, sizeof(length));
    if (bytes_written != sizeof(length)) {
        std::cerr << "Failed to write message length: " << strerror(errno) << std::endl;
        return;
    }

    // Send message
    size_t total_written = 0;
    while (total_written < length) {
        const ssize_t n = write(client_fd, msg_str.data() + total_written, length - total_written);
        if (n <= 0) {
            std::cerr << "Failed to write message data: " << strerror(errno) << std::endl;
            return;
        }
        total_written += static_cast<size_t>(n);
    }
}

void SocketServer::close_connection() noexcept {
    close_fd(client_fd);
    close_fd(server_fd);
    if (mode == Mode::Unix && !socket_path.empty()) {
        unlink(socket_path.c_str());
    }
}
