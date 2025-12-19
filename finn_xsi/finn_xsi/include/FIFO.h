#ifndef FIFO_H
#define FIFO_H

#include <cstdint>
#include <limits>

class FIFO {
    uint64_t maxUtil = 0;
    uint64_t currentUtil = 0;
    uint64_t maxSize = 0;
    uint64_t nextUtil = 0;

     public:
    FIFO(uint64_t size = std::numeric_limits<uint64_t>::max());
    ~FIFO();

    void update(bool incomingValid, bool incomingReady);
    void toggleClock();
    bool isInputReady() const;
    bool isOutputValid() const;
    bool isEmpty() const;
    void reset(uint64_t size = std::numeric_limits<uint64_t>::max());
    void setMaxSize(const uint64_t size);
    uint64_t getMaxSize() const;
    uint64_t getSpaceLeft() const;
    uint64_t getMaxUtil() const;
    void increaseCounter(const uint64_t count);

    // NOTE: User needs to ensure proper ordering. No runtime enforcement of order.
    void tryPush(bool incomingValid);
    void tryPop(bool incomingReady);
    uint64_t size() const;
};

#endif /* FIFO_H */
