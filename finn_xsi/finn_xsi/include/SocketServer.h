#ifndef SOCKET_SERVER_H
#define SOCKET_SERVER_H

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>

using json = nlohmann::ordered_json;

class SocketServer {
     private:
    int server_fd{-1};
    int client_fd{-1};
    std::string socket_path;

    void close_fd(int& fd) noexcept;

     public:
    explicit SocketServer(std::string_view path);
    ~SocketServer();

    // Disable copy construction and assignment
    SocketServer(const SocketServer&) = delete;
    SocketServer& operator=(const SocketServer&) = delete;

    // Enable move semantics
    SocketServer(SocketServer&& other) noexcept;
    SocketServer& operator=(SocketServer&& other) noexcept;

    // Returns std::nullopt on success, error message on failure
    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] std::optional<json> receive_message();
    void send_message(const json& message);
    void close_connection() noexcept;

    [[nodiscard]] bool is_connected() const noexcept { return client_fd >= 0; }
};

#endif  // SOCKET_SERVER_H
