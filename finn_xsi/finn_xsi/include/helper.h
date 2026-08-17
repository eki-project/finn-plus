#ifndef HELPER_H_
#define HELPER_H_

#include <array>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>

constexpr std::array<char, 4> XZ10 = {'0', '1', 'Z', 'X'};
constexpr std::array<char, 16> HEX = {'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

struct StreamDescriptor {
  std::string_view name;
  std::size_t job_size;
  // // Next job can only start this many clock ticks after start of predecessor.
  // std::size_t job_ticks;
};

#ifdef NDEBUG
[[maybe_unused]] inline void debug([[maybe_unused]] std::string_view s) {}
#else
inline void debug(std::string_view s) { std::cout << "log [DBG] " << s << "\n"; }
#endif

inline std::string timestamp_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

inline std::string& log_prefix() {
    static std::string prefix;
    return prefix;
}

#define FINN_LOG(x)                                                                     \
    do {                                                                                \
        std::cout << "[" << timestamp_now() << "]" << log_prefix() << " " << x << "\n"; \
        std::cout.flush();                                                              \
    } while (0)

#endif /* HELPER_H_ */
