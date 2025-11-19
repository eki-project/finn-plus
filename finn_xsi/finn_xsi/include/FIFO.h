#ifndef FIFO_H
#define FIFO_H

#include <limits>
#include <optional>

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
