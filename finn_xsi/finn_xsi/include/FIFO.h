#ifndef FIFO_H
#define FIFO_H

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
/*
class FIFO {
    std::size_t currentUtil = 0;
    std::size_t maxUtil = 0;
    std::size_t maxSize = 0;

     public:

    FIFO(std::size_t maxSize = std::numeric_limits<std::size_t>::max());
    ~FIFO();

    // Add FIFO methods and members as needed
    bool isEmpty() const;
    void tryPushPop(bool incomingValid, bool successorReady);
    void tryPop(bool successorReady);
    bool isOutputValid() const;
    bool isInputReady() const;
    void tryPush(bool incomingValid);
    void reset();
    void setMaxSize(std::size_t newSize);
};*/

// TODO: switch int to std::size_t

class FIFO {
    int maxUtil = 0;
    int currentUtil = 0;
    int maxSize = 0;
    int nextUtil = 0;

    public:
    FIFO(int size = std::numeric_limits<int>::max());
    ~FIFO();

    void update(bool incomingValid, bool outgoingReady);
    void toggleClock();
    bool isInputReady() const;
    bool isOutputValid() const;
    bool isEmpty() const;
    void reset(std::optional<int> size = std::nullopt);
    void setMaxSize(int size);
};

#endif /* FIFO_H */
