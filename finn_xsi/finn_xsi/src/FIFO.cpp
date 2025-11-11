#include <FIFO.h>

#include <algorithm>

FIFO::FIFO(std::size_t p_max_size) : util(0), max_util(0), max_size(p_max_size) {}

FIFO::~FIFO() {}

bool FIFO::is_valid() {
    if (sucReady && util > 0) {
        --util;
        return true;
    }
    return util > 0;
}

void FIFO::ready(bool ready) {
    sucReady = ready;
    readyCtr = ready ? readyCtr + 1 : readyCtr;
}

bool FIFO::is_ready() const { return util < max_size; }

void FIFO::write(bool valid) {
    if (valid && util < max_size) {
        max_util = std::max(max_util, ++util);
    }
}

std::size_t FIFO::get_largest_occupation() const { return max_util; }

void FIFO::reset() {
    util = 0;
    max_util = 0;
    sucReady = false;
}
