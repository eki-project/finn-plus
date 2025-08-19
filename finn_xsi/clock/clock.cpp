#include "clock.h"
#include <iostream>
#include "../xsi_finn.hpp"

using namespace xsi;

Clock::Clock(xsi::Design& design) : design(design) {
    // Find Global Control & Run Startup Sequence
	Port *const  clk   = design.getPort("ap_clk");
	Port *const  clk2x = design.getPort("ap_clk2x");
	if(!clk) {
		throw std::runtime_error("No clock found on the design.");
	}
	cycle = clk2x?
		std::function<void(bool)>([&design, clk, clk2x](bool const  up) mutable {
			clk->set(up).write_back();
			clk2x->set(1).write_back();
			design.run(5);
			clk2x->set(0).write_back();
			design.run(5);
		}) :
		std::function<void(bool)>([&design, clk](bool const  up) mutable {
			clk->set(up).write_back();
			design.run(5);
		});
}

void Clock::toggle_clk() {
    cycle(1);
    cycle(0);
}
