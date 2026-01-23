#ifndef FIFO_H
#define FIFO_H

#include <CommunicationChannel.hpp>
#include <cstdint>
#include <limits>
#include <stop_token>

class FIFO : public CommunicationChannel {
    uint64_t maxUtil = 0;
    uint64_t currentUtil = 0;
    uint64_t maxSize = 0;
    uint64_t nextUtil = 0;

     public:
    FIFO(uint64_t size = std::numeric_limits<uint64_t>::max());
    ~FIFO();

    void update(bool incomingValid, bool incomingReady);
    void toggleClock();
    virtual bool getInputReady(std::stop_token stoken = {}) noexcept override;
    virtual bool getOutputValid(std::stop_token stoken = {}) noexcept override;
    bool isEmpty() const;
    void reset(uint64_t size = std::numeric_limits<uint64_t>::max());
    void setMaxSize(const uint64_t size);
    uint64_t getMaxSize() const;
    uint64_t getSpaceLeft() const;
    uint64_t getMaxUtil() const;
    void increaseCounter(const uint64_t count);

    // NOTE: User needs to ensure proper ordering. No runtime enforcement of order.
    virtual void setInputValid(bool incomingValid, std::stop_token stoken = {}) override;
    virtual void setOutputReady(bool incomingReady, std::stop_token stoken = {}) override;
    uint64_t size() const;
};

#endif /* FIFO_H */
