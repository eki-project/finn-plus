#include <FIFO.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

FIFO::FIFO(uint64_t size) : maxSize(size) {}
FIFO::~FIFO() {}

/// Prepare update for the next clock cycle.
/// This models Q_srl behavior where:
/// - When empty: only accepts input (ignores output ready), transitions to size 1
/// - When non-empty: can consume, produce, or both
/// With bounded maxSize, this models a real FIFO with backpressure.
void FIFO::update(bool incomingValid, bool incomingReady) {
    // When empty: only push if valid (ignoring ready)
    // When non-empty: push if valid AND space available
    uint64_t canPush = incomingValid & (currentUtil < maxSize);

    // Q_srl behavior: when empty, only check input valid (ignore output ready)
    // Only pop if was non-empty at start AND output ready
    uint64_t canPop = incomingReady & (currentUtil != 0);

    nextUtil = nextUtil + canPush - canPop;
}

/// Toggle the clock cycle, and update the previously set values.
/// nextUtil is guaranteed to be in [0, maxSize] by all operations.
/// Returns false if a first valid signal was expected, but has not been observed.
bool FIFO::toggleClock() {
    currentUtil = nextUtil;
    maxUtil = std::max(maxUtil, currentUtil);
    nextUtil = currentUtil;
    cyclesUntilExpectedFirstValid -= static_cast<uint64_t>(static_cast<bool>(cyclesUntilExpectedFirstValid) & !static_cast<bool>(maxUtil));  // Underflow-safe decrement
    return (cyclesUntilExpectedFirstValid == 0) & (maxUtil == 0);
}

/// Return whether the FIFO can accept inputs (for the current utilization)
/// Uses nextUtil (post-push state) so that ready correctly reflects capacity
/// after any push already committed this cycle, preventing AXI-S violations.
bool FIFO::getInputReady([[maybe_unused]] std::stop_token stoken) noexcept { return nextUtil < maxSize; }

/// Return whether the FIFO can output values (for the current utilization)
bool FIFO::getOutputValid([[maybe_unused]] std::stop_token stoken) noexcept { return currentUtil > 0; }

/// Return whether the FIFO is empty (for the current utilization)
bool FIFO::isEmpty() const { return currentUtil == 0; }

/// Reset the FIFOs internal state. If size is given, also set maxSize,
/// otherwise keep it.
void FIFO::reset(uint64_t size) {
    currentUtil = 0;
    maxUtil = 0;
    maxSize = size;
    nextUtil = 0;
    cyclesUntilExpectedFirstValid = std::numeric_limits<uint64_t>::max();
}

void FIFO::setCyclesUntilExpectedFirstValid(uint64_t cycles) {
    cyclesUntilExpectedFirstValid = cycles;
    initialCyclesUntilExpectedFirstValid = cycles;
    std::cout << "FIFO set to expect first valid after " << cycles << " cycles" << std::endl;
}

uint64_t FIFO::getCyclesUntilFirstValid() const { return initialCyclesUntilExpectedFirstValid - cyclesUntilExpectedFirstValid; }

/// Set the FIFOs max size
void FIFO::setMaxSize(const uint64_t size) { maxSize = size; }

uint64_t FIFO::getMaxSize() const { return maxSize; }

uint64_t FIFO::getSpaceLeft() const { return maxSize - currentUtil; }

uint64_t FIFO::getMaxUtil() const { return maxUtil; }

void FIFO::increaseCounter(const uint64_t count) {
    // Branchless: compute new value and saturate at maxSize
    uint64_t newUtil = nextUtil + count;
    uint64_t overflow = newUtil > maxSize;
    nextUtil = overflow ? maxSize : newUtil;
}

/// If incomingValid is true and FIFO has space, increment nextUtil
/// Matches Q_srl: when empty, always accepts input
/// When using tryPush/tryPop separately, ALWAYS call tryPush BEFORE tryPop!
void FIFO::setInputValid(bool incomingValid, [[maybe_unused]] std::stop_token stoken) {
    // When empty: accept input unconditionally (like Q_srl state_empty)
    // When non-empty: accept if space available
    nextUtil += incomingValid & (nextUtil < maxSize);
}

/// If incomingReady is true and FIFO has data, decrement nextUtil
/// Matches Q_srl: only pops if data available
/// When using tryPush/tryPop separately, ALWAYS call tryPush BEFORE tryPop!
/// Note: If FIFO was empty and tryPush just added data, tryPop will NOT pop it
/// (matching Q_srl where state_empty ignores output ready)
void FIFO::setOutputReady(bool incomingReady, [[maybe_unused]] std::stop_token stoken) {
    // Check currentUtil (state at cycle start) not nextUtil (after tryPush)
    // This ensures empty->tryPush->tryPop results in size=1, matching Q_srl
    nextUtil -= incomingReady & (currentUtil > 0);
}

/// Return the current number of elements in the FIFO
uint64_t FIFO::size() const { return currentUtil; }
