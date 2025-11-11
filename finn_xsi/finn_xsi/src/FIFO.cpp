#include <FIFO.h>
#include <algorithm>

FIFO::FIFO(std::size_t max_size) : currentUtil(0), maxUtil(0), maxSize(max_size) {}

FIFO::~FIFO() {}

bool FIFO::isValid() {
    if (sucReady && currentUtil > 0) {
        --currentUtil;
        return true;
    }
    return false;
}

void FIFO::ready(bool ready) { sucReady = ready; }

bool FIFO::isReady() const { return currentUtil < maxSize; }

void FIFO::write(bool valid) {
    if (valid && currentUtil < maxSize) {
        maxUtil = std::max(maxUtil, ++currentUtil);
    }
}

std::size_t FIFO::getLargestOccupation() const { return maxUtil; }

void FIFO::setMaxSize(size_t newSize) {
    maxSize = newSize;
}

void FIFO::reset() {
    currentUtil = 0;
    maxUtil = 0;
    sucReady = false;
}
