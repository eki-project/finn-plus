#ifndef COMMUNICATIONCHANNEL
#define COMMUNICATIONCHANNEL

#include <concepts>
#include <stop_token>
#include <iostream>

template<typename T>
concept ChannelInterface = requires(T t, bool b, std::stop_token stoken) {
    { t.getOutputValid(stoken) } -> std::same_as<bool>;
    { t.setInputValid(b, stoken) } -> std::same_as<void>;
    { t.getInputReady(stoken) } -> std::same_as<bool>;
    { t.setOutputReady(b, stoken) } -> std::same_as<void>;
};

class CommunicationChannel {
    // Function pointers for downstream object methods
    bool (*downstreamGetInputReadyFn)(void*, std::stop_token) = nullptr;
    void (*downstreamSetInputValidFn)(void*, bool, std::stop_token) = nullptr;

    void* downstreamObj = nullptr;

     protected:
    // Derived classes call this to register their own methods
    template<ChannelInterface Derived>
    void registerSelfAs() {
        // This is intentionally empty - we call methods directly on 'this'
        // The template just ensures Derived implements ChannelInterface
    }

     public:
    template<ChannelInterface Derived>
    void connectDownstream(Derived& downstreamPartner) {
        this->downstreamObj = &downstreamPartner;

        // Store function pointers for calling the DOWNSTREAM object's methods
        downstreamGetInputReadyFn = [](void* obj, std::stop_token stoken) -> bool { return static_cast<Derived*>(obj)->getInputReady(stoken); };
        downstreamSetInputValidFn = [](void* obj, bool v, std::stop_token stoken) { static_cast<Derived*>(obj)->setInputValid(v, stoken); };
    }

    // Mark as inline and noexcept for better optimization
    inline void exchangeDataDownstream(std::stop_token stoken = {}) noexcept {
        // Call methods on THIS object directly (non-virtual, resolved at compile time)
        bool valid = this->getOutputValid(stoken);
        // Call downstream object's methods via function pointers
        downstreamSetInputValidFn(downstreamObj, valid, stoken);
        bool ready = downstreamGetInputReadyFn(downstreamObj, stoken);
        // Call method on THIS object directly
        this->setOutputReady(ready, stoken);
    }

    virtual bool getOutputValid([[maybe_unused]] std::stop_token stoken = {}) = 0;
    virtual void setInputValid([[maybe_unused]] bool v, [[maybe_unused]] std::stop_token stoken = {}) = 0;
    virtual bool getInputReady([[maybe_unused]] std::stop_token stoken = {}) = 0;
    virtual void setOutputReady([[maybe_unused]] bool r, [[maybe_unused]] std::stop_token stoken = {}) = 0;

    virtual ~CommunicationChannel() = default;
};

// Example usage:
// class LayerA : public CommunicationChannel {
// public:
//     bool getOutputValid(std::stop_token stoken = {}) { /* ... */ }
//     void setInputValid(bool v, std::stop_token stoken = {}) { /* ... */ }
//     bool getInputReady(std::stop_token stoken = {}) { /* ... */ }
//     void setOutputReady(bool r, std::stop_token stoken = {}) { /* ... */ }
// };
//
// LayerA a;
// LayerB b;
// a.connectDownstream(b);
// a.exchangeDataDownstream(); // or with stop_token: a.exchangeDataDownstream(stoken);

#endif /* COMMUNICATIONCHANNEL */
