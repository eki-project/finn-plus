#ifndef CLOCK
#define CLOCK

#include <functional>

namespace xsi {
    class Design;
}

class Clock {
    xsi::Design& design;
    Clock(xsi::Design& design);

    Clock(Clock const&) = delete;
    Clock& operator=(Clock const&) = delete;
public:
    ~Clock() = default;

    std::function<void(bool)>  cycle;

    static Clock& initClock(xsi::Design& design){
        static Clock clk(design);
        return clk;
    }

    void toggle_clk();

};

#endif /* CLOCK */
