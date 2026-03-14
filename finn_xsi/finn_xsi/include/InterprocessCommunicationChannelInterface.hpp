#ifndef INTERPROCESSCOMMUNICATIONCHANNELINTERFACE
#define INTERPROCESSCOMMUNICATIONCHANNELINTERFACE

#include <CommunicationChannel.hpp>
#include <InterprocessCommunicationChannel.hpp>
#include <stop_token>

template<bool IsSender>
class InterprocessCommunicationChannelInterface : public CommunicationChannel {
    struct Forward {
        bool valid;
    };

    struct Backward {
        bool ready;
    };

    InterprocessCommunicationChannel<Forward, Backward, IsSender> channel;
    Backward lastResponse;

     public:
    // Default constructor
    InterprocessCommunicationChannelInterface() = default;

    // Constructor with shared memory name
    explicit InterprocessCommunicationChannelInterface(const std::string& shmName) : channel(shmName), lastResponse{false} {}

    // Delete copy operations
    InterprocessCommunicationChannelInterface(const InterprocessCommunicationChannelInterface&) = delete;
    InterprocessCommunicationChannelInterface& operator=(const InterprocessCommunicationChannelInterface&) = delete;

    // Move constructor
    InterprocessCommunicationChannelInterface(InterprocessCommunicationChannelInterface&& other) noexcept = default;

    // Move assignment operator
    InterprocessCommunicationChannelInterface& operator=(InterprocessCommunicationChannelInterface&& other) noexcept = default;

    virtual bool getInputReady([[maybe_unused]] std::stop_token stoken = {}) override {
        if constexpr (!IsSender) {
            throw std::runtime_error("getInputReady can only be called on sender instances.");
        } else {
            return lastResponse.ready;
        }

    }
    virtual bool getOutputValid(std::stop_token stoken = {}) override {
        if constexpr (IsSender) {
            throw std::runtime_error("getOutputValid can only be called on receiver instances.");
        } else {
            return channel.receive_request(stoken).valid;
        }
    }

    virtual void setInputValid(bool incomingValid, std::stop_token stoken = {}) override {
        if constexpr (!IsSender) {
            throw std::runtime_error("setInputValid can only be called on sender instances.");
        } else {
            lastResponse = channel.send_request(Forward{incomingValid}, stoken);
        }
    }
    virtual void setOutputReady(bool incomingReady, [[maybe_unused]] std::stop_token stoken = {}) override {
        if constexpr (IsSender) {
            throw std::runtime_error("setOutputReady can only be called on receiver instances.");
        } else {
            channel.send_response(Backward{incomingReady});
        }
    }

    virtual ~InterprocessCommunicationChannelInterface() = default;
};

#endif /* INTERPROCESSCOMMUNICATIONCHANNELINTERFACE */
