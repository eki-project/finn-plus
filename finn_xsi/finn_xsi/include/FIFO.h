#ifndef FIFO_H
#define FIFO_H

#include <cstddef>
#include <limits>

class FIFO {
    std::size_t currentUtil = 0;
    std::size_t maxUtil = 0;
    std::size_t maxSize = 0;
    bool sucReady = false;

     public:
    FIFO(std::size_t maxSize = std::numeric_limits<std::size_t>::max());
    ~FIFO();

    // Add FIFO methods and members as needed
    bool isValid();
    void ready(bool ready);
    bool isReady() const;
    void write(bool valid);
    std::size_t getLargestOccupation() const;
    void reset();
    void setMaxSize(std::size_t newSize);
};

#endif /* FIFO_H */
