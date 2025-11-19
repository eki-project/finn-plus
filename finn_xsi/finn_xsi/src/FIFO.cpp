#include <FIFO.h>
#include <algorithm>
#include <optional>

/*
FIFO::FIFO(std::size_t max_size) : currentUtil(0), maxUtil(0), maxSize(max_size) {}

FIFO::~FIFO() {}

bool FIFO::isEmpty() const {
    return currentUtil == 0;
}

void FIFO::tryPushPop(bool incomingValid, bool successorReady) {
    int diff = static_cast<int>(incomingValid) - static_cast<int>(successorReady);
    if (diff == 1 && currentUtil < maxSize) {
        ++currentUtil;
        maxUtil = std::max(maxUtil, ++currentUtil);
    } else if (diff == -1 && currentUtil > 0) {
        --currentUtil;
    }
    // Otherwise do nothing
}

void FIFO::tryPop(bool successorReady) {
    if (successorReady && currentUtil > 0) {
        --currentUtil;
    }
}

bool FIFO::isOutputValid() const { return currentUtil > 0; }

bool FIFO::isInputReady() const { return currentUtil < maxSize; }

void FIFO::tryPush(bool incomingValid) {
    if (incomingValid && currentUtil < maxSize) {
        maxUtil = std::max(maxUtil, ++currentUtil);
    }
}

void FIFO::setMaxSize(size_t newSize) {
    maxSize = newSize;
}

void FIFO::reset() {
    currentUtil = 0;
    maxUtil = 0;
}
*/

FIFO::FIFO(int size) : maxSize(size) {}
FIFO::~FIFO() {}

/// Prepare update for the next clock cycle.
void FIFO::update(bool incomingValid, bool outgoingReady) {
    int diff = static_cast<int>(incomingValid) - static_cast<int>(outgoingReady);
    if ((currentUtil > 0 && currentUtil < maxSize) || (currentUtil == 0 && diff > 0) || (currentUtil == maxSize && diff < 0)) {
        nextUtil = currentUtil += diff;
    }
}

/// Toggle the clock cycle, and update the previously set values.
void FIFO::toggleClock() {
    currentUtil = nextUtil;
}

/// Return whether the FIFO can accept inputs (for the current utilization)
bool FIFO::isInputReady() const { return currentUtil < maxSize; }

/// Return whether the FIFO can output values (for the current utilization)
bool FIFO::isOutputValid() const { return currentUtil > 0; }

/// Return whether the FIFO is empty (for the current utilization)
bool FIFO::isEmpty() const { return currentUtil == 0; }

/// Reset the FIFOs internal state. If size is given, also set maxSize, otherwise keep it.
void FIFO::reset(std::optional<int> size) {
    currentUtil = 0;
    maxUtil = 0;
    if (size) {
        maxSize = *size;
    }
    nextUtil = 0;
}

/// Set the FIFOs max size
void FIFO::setMaxSize(int size) {
    maxSize = size;
}
