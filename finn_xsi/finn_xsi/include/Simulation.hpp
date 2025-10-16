#ifndef SIMULATION
#define SIMULATION
#include <AXIS_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <SharedLibrary.h>
#include <helper.h>

template< size_t IStreamsSize, size_t OStreamsSize >
class Simulation {
public:
  xsi::Kernel kernel;
  xsi::Design top;
  std::array<S_AXIS_Control, IStreamsSize> istreams;
  std::array<M_AXIS_Control, OStreamsSize> ostreams;
  Clock clk;

  void clearPorts() noexcept {
    // Clear all input ports
    for (xsi::Port &p : top.ports()) {
      if (p.isInput()) {
        p.clear().write_back();
      }
    }
  }

  void reset() noexcept {
    xsi::Port &rst_n = top.getPort("ap_rst_n");
    // Reset all Inputs, Wait for Reset Period
    rst_n.set(0).write_back();
    for (unsigned i = 0; i < 16; i++) {
      clk.toggle_clk();
    }
    rst_n.set(1).write_back();
  }

  Simulation(const std::string &kernel_lib, const std::string &design_lib,
             const char *xsim_log_file, const char *trace_file, std::array<StreamDescriptor, IStreamsSize> istream_descs,
             std::array<StreamDescriptor, OStreamsSize> ostream_descs)
      : kernel(kernel_lib), top(kernel, design_lib, xsim_log_file, trace_file),
        clk(top) {
    if (trace_file) {
      top.trace_all();
    }

    // Find I/O Streams and initialize their Status
    for (size_t i = 0; i < istream_descs.size(); ++i) {
      istreams[i] =
          S_AXIS_Control{top, clk, std::data(istream_descs)[i].job_size,
                         std::data(istream_descs)[i].job_ticks,
                         std::data(istream_descs)[i].name};
    }
    for (size_t i = 0; i < ostream_descs.size(); ++i) {
      ostreams[i] =
          M_AXIS_Control{top, clk, std::data(ostream_descs)[i].job_size,
                         std::data(ostream_descs)[i].name};
    }

    // Find Global Control & Run Startup Sequence
    clearPorts();
    reset();

    // Make all Inputs valid & all Outputs ready
    for (auto &&s : istreams) {
      s.valid();
    }
    for (auto &&s : ostreams) {
      s.ready();
    }
  }
};

#endif /* SIMULATION */
