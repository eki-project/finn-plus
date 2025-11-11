#ifndef FIFO_H
#define FIFO_H

#include <cstddef>
#include <limits>

class FIFO {
    std::size_t util = 0;
    std::size_t max_util = 0;
    std::size_t max_size = 0;
    bool sucReady = false;

    std::size_t readyCtr = 0;

     public:
    FIFO(std::size_t max_size = std::numeric_limits<std::size_t>::max());
    ~FIFO();

    // Add FIFO methods and members as needed
    bool is_valid();
    void ready(bool ready);
    bool is_ready() const;
    void write(bool valid);
    std::size_t get_largest_occupation() const;
    std::size_t getReadyCtr() const { return readyCtr; }
    void reset();
};

#endif /* FIFO_H */
