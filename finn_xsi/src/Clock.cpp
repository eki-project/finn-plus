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
    cycle = clk2x ? std::function<void(bool)>([&des, &clk, clk2x](bool const up) mutable {
        clk.set(up).write_back();
        clk2x->set(1).write_back();
        des.run(5);
        clk2x->set(0).write_back();
        des.run(5);
    })
                  : std::function<void(bool)>([&des, &clk](bool const up) mutable {
                        clk.set(up).write_back();
                        des.run(5);
                    });
}

void Clock::toggle_clk() noexcept {
    cycle(1);
    cycle(0);
}
