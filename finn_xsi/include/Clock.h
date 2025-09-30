#ifndef CLOCK
#define CLOCK

#include <functional>

// Fwd declarations
namespace xsi {
    class Design;
}

struct simDesc;

class Clock {
    xsi::Design& design;

    Clock(Clock const&) = delete;
    Clock& operator=(Clock const&) = delete;
    Clock(xsi::Design& design);
    friend simDesc;

     public:
    Clock(Clock&&) noexcept = default;
    Clock& operator=(Clock&&) noexcept = default;
    ~Clock() noexcept = default;

    std::function<void(bool)> cycle;


    void toggle_clk() noexcept;
};

#endif /* CLOCK */
