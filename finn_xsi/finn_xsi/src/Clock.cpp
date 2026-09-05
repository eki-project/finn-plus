#include <Clock.h>
#include <Design.h>
#include <Port.h>

using namespace xsi;

Clock::Clock(xsi::Design& des) : design(des) {
    // Find Global Control & Run Startup Sequence
    Port& clk = des.getPort("ap_clk");
    auto ports = des.ports();

    Port* clk2x = nullptr;
    for (auto&& p : ports) {
        if (p.name() == std::string("ap_clk2x")) {
            clk2x = &p;
            break;
        }
    }
    clkHigh = clk2x ? std::function<void()>([&des, &clk, clk2x]() mutable {
        des.run(1);
        clk.set(1).write_back();
        clk2x->set(1).write_back();
        des.run(1);
    }) : std::function<void()>([&des, &clk]() mutable {
        des.run(1);
        clk.set(1).write_back();
        des.run(1);
    });
    clkLow = clk2x ? std::function<void()>([&des, &clk, clk2x]() mutable {
        des.run(2499);
        clk2x->set(0).write_back();
        des.run(2500);
        clk.set(0).write_back();
        clk2x->set(1).write_back();
        des.run(2500);
        clk2x->set(0).write_back();
        des.run(2499);

    }) : std::function<void()>([&des, &clk]() mutable {
        des.run(4999);
        clk.set(0).write_back();
        des.run(4999);
    });
    // cycle = clk2x ? std::function<void(bool)>([&des, &clk, clk2x](bool const up) mutable {
    //     clk.set(up).write_back();
    //     clk2x->set(1).write_back();
    //     des.run(5000);
    //     clk2x->set(0).write_back();
    //     des.run(5000);
    // })
    //               : std::function<void(bool)>([&des, &clk](bool const up) mutable {
    //                     clk.set(up).write_back();
    //                     des.run(5000);
    //                 });
}

void Clock::toggleClk() noexcept {
    clkHigh();
    clkLow();
}

void Clock::clockHigh() noexcept {
    clkHigh();
}

void Clock::clockLow() noexcept {
    clkLow();
}
