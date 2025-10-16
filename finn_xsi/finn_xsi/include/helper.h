#ifndef HELPER_H_
#define HELPER_H_

#include <array>
#include <string>

constexpr std::array<char, 4> XZ10 = {'0', '1', 'Z', 'X'};
constexpr std::array<char, 16> HEX = {'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

struct StreamDescriptor {
  std::string name;
  std::size_t job_size;
  // Next job can only start this many clock ticks after start of predecessor.
  std::size_t job_ticks;
};

#endif /* HELPER_H_ */
