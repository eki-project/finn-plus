#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "axi_control/s_axi_control.h"
#include "axis_control/axis_control.h"
#include "clock/clock.h"
#include "rtlsim_config.hpp"
#include "xsi_finn.hpp"

void clearPorts(xsi::Design &top) {
  // Clear all input ports
  for (xsi::Port &p : top.ports()) {
    if (p.isInput()) {
      p.clear().write_back();
    }
  }
}

void reset(xsi::Design &top) {
  xsi::Port *const rst_n = top.getPort("ap_rst_n");
  Clock &clk = Clock::initClock(top);
  if (!rst_n) {
    std::cerr << "No reset port found in design." << std::endl;
    return;
  }
  // Reset all Inputs, Wait for Reset Period
  rst_n->set(0).write_back();
  if (rst_n) {
    for (unsigned i = 0; i < 32; i++) {
      clk.toggle_clk();
    }
    rst_n->set(1).write_back();
    for (unsigned i = 0; i < 16; i++) {
      clk.toggle_clk();
    }
  }
}

std::unordered_map<std::string, int> fifo_register_map = {
    {"mode", 0x10},
    {"depth", 0x18},
    {"occupancy", 0x20},
    {"max_occupancy", 0x28}};

void configure_fifo(S_AXI_Control &fifo, const int mode, uint32_t depth) {
  fifo.write_register(fifo_register_map["mode"], mode);
  std::cout << "Mode written" << std::endl;
  fifo.write_register(fifo_register_map["depth"], depth);
}

// Helper function to process input streams with deferred writes
void processInputStream(
    S_AXIS_Control &stream, size_t &iters,
    std::vector<std::reference_wrapper<xsi::Port>> &deferredWrites,
    std::deque<size_t> &inflightTimestamps) {
  const bool isValid = stream.is_valid();

  // Skip if valid but not ready
  if (isValid && !stream.is_ready()) {
    return;
  }

  // Track successful transactions
  if (isValid) {
    stream.job_txns++;
  }

  const bool shouldProcessJob =
      (stream.job_txns < stream.job_size) || (iters >= stream.await_iter);

  // Handle throttling
  if (shouldProcessJob) {
    if (!isValid) {
      deferredWrites.emplace_back(stream.set_valid(true));
    }

    // Reset job when completed
    if (stream.job_txns == stream.job_size) {
      stream.job_txns = 0;
      stream.await_iter = iters + stream.job_ticks;
      inflightTimestamps.push_front(iters);
    }
  } else if (isValid) {
    deferredWrites.emplace_back(stream.set_valid(false));
  }
}

// Helper function to process output streams
bool processOutputStream(M_AXIS_Control &stream, size_t &iters,
                         std::deque<size_t> &inflightTimestamps,
                         size_t &completedMaps) {
  if (!stream.is_ready() || !stream.is_valid()) {
    return false;
  }

  const size_t totalTransactions = ++stream.total_txns;

  // Track first completion
  if (totalTransactions == stream.job_size) {
    stream.first_complete = iters;
  }

  // Track job completion and intervals
  if (++stream.job_txns == stream.job_size) {
    stream.interval = iters - stream.last_complete;
    stream.last_complete = iters;
    stream.job_txns = 0;
    stream.latency = iters - inflightTimestamps.back();
    inflightTimestamps.pop_back(); // Remove the oldest timestamp
    stream.min_latency = std::min(stream.min_latency, stream.latency);
    ++completedMaps;
  }

  return true;
}

bool runForFeaturemaps(size_t featuremaps, Clock &clk,
                       std::vector<S_AXIS_Control> &istreams,
                       std::vector<M_AXIS_Control> &ostreams,
                       std::deque<size_t> &inflightTimestamps, size_t &iters) {
  // Enter Simulation Loop and track Progress
  auto const begin = std::chrono::steady_clock::now();
  std::vector<std::reference_wrapper<xsi::Port>> to_write;
  size_t completedMaps = 0;
  size_t timeout = 0;
  while (true) {

    //-------------------------------------------------------------------
    // Clock down - then read signal updates from design
    clk.cycle(0);

    //-------------------------------------------------------------------
    // Process input streams

    for (auto &stream : istreams) {
      processInputStream(stream, iters, to_write, inflightTimestamps);
    }

    //-------------------------------------------------------------------
    // Process output streams

    bool hasActiveOutput = false;
    for (auto &stream : ostreams) {
      if (processOutputStream(stream, iters, inflightTimestamps,
                              completedMaps)) {
        hasActiveOutput = true;
      }
    }
    timeout = hasActiveOutput ? 0 : timeout + 1;

    //-------------------------------------------------------------------
    // Clock up - then write signal updates back to design
    clk.cycle(1);

    // Write back Ports with registered updates
    for (xsi::Port &p : to_write)
      p.write_back();
    to_write.clear();

    ++iters;

    //if(iters % 25000 == 0){
      std::cout << "Iteration: " << iters
              << ", Completed Maps: " << completedMaps
              << ", Inflight Timestamps Size: " << inflightTimestamps.size()
              << ", in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "ms"
              << std::endl;
    //}

    if (completedMaps == featuremaps) {
      break;
    }
    // Check for timeout
    if (timeout > max_iters) {
      return false;
    }
  }
  return true;
}

size_t computeMovingAverage(const size_t oldAvg, const size_t newValue,
                            const size_t count) {
  return (oldAvg * (count - 1) + newValue) / count;
}

std::optional<std::tuple<size_t, size_t>>
runToStableState(Clock &clk, std::vector<S_AXIS_Control> &istreams,
                 std::vector<M_AXIS_Control> &ostreams,
                 std::deque<size_t> &inflightTimestamps, size_t &iters) {
  // size_t avgLatency = 0;
  size_t avgMinLatency = 0;
  size_t avgInterval = 0;
  size_t runs = 0;

  while (true) {
    if (!runForFeaturemaps(1, clk, istreams, ostreams, inflightTimestamps,
                           iters)) {
      return std::nullopt;
    }

    size_t newAvgMinLatency =
        computeMovingAverage(avgMinLatency, ostreams[0].min_latency, ++runs);
    size_t newAvgInterval =
        computeMovingAverage(avgInterval, ostreams[0].interval, runs);

    bool changed =
        (avgMinLatency != newAvgMinLatency) || (avgInterval != newAvgInterval);

    avgMinLatency = newAvgMinLatency;
    avgInterval = newAvgInterval;

    if (!changed) {
      break;
    }
  }
  // Avg Latency and Interval are here equivalent to the output stream's
  // metrics, but easier to retrieve
  return std::make_tuple(avgMinLatency, avgInterval);
}

std::tuple<size_t, size_t>
determineStartDepth(xsi::Design &top, Clock &clk,
                    std::vector<S_AXIS_Control> &istreams,
                    std::vector<M_AXIS_Control> &ostreams,
                    std::deque<size_t> &inflightTimestamps, size_t &iters,
                    std::vector<S_AXI_Control> &fifos) {
  int start_depth = 3612672;
  int last_start_depth = 3612672;
  uint32_t last_interval = 0;
  uint32_t min_latency = 0;

  while (true) {
    std::cout << "Starting testing start_depth: " << start_depth << std::endl;

    reset(top);
    std::cout << "After reset" << std::endl;
    for (auto &fifo : fifos) {
      configure_fifo(fifo, 1, start_depth);
    }

    std::cout << "Configuring FIFOs done." << std::endl;

    if (auto ret = runToStableState(clk, istreams, ostreams, inflightTimestamps,
                                    iters);
        ret) {
      auto &&[latency, interval] = *ret;
      if (interval > 0 && interval == last_interval) {
        start_depth = last_start_depth;
        break;
      }
      std::cout << "Testing start_depth: " << start_depth
                << ", latency: " << latency << ", interval: " << interval
                << std::endl;
      last_interval = interval;
      min_latency = latency;
    }

    last_start_depth = start_depth;
    start_depth *= 2; // Double the start depth

    if (start_depth > 1000000) {
      throw std::runtime_error(
          "Couldn't find a working start depth, please set manually!");
    }
  }

  return std::make_tuple(start_depth, last_interval);
}

std::vector<size_t> sizeIteratively(size_t start_size, size_t interval,
                                    Clock &clk, xsi::Design &top,
                                    std::vector<S_AXI_Control> &fifos,
                                    std::vector<S_AXIS_Control> &istreams,
                                    std::vector<M_AXIS_Control> &ostreams,
                                    std::deque<size_t> &inflightTimestamps,
                                    size_t &iters) {
  std::vector<bool> minimizedFifo(fifos.size(), false);
  std::vector<size_t> fifo_sizes(fifos.size(), start_size);

  while (!std::all_of(minimizedFifo.begin(), minimizedFifo.end(),
                      [](bool v) { return v; })) {
    for (size_t i = 0; i < fifos.size(); ++i) {
      if (!minimizedFifo[i]) {
        // Try to minimize this FIFO
        size_t oldFifoSize = fifo_sizes[i];
        fifo_sizes[i] = std::max(oldFifoSize / 2, size_t(1));

        reset(top);
        for (auto &&fifo : fifos) {
          configure_fifo(fifo, 1, fifo_sizes[i]);
        }

        if (auto ret = runToStableState(clk, istreams, ostreams,
                                        inflightTimestamps, iters);
            ret) {
          auto &&[latency, currInterval] = *ret;
          if (currInterval == 0 || currInterval > interval) {
            // Performance drop
            // Revert depth reduction and mark FIFO as minimized
            fifo_sizes[i] = oldFifoSize;
            minimizedFifo[i] = true;
          }
        } else {
          // Timeout
          // Revert depth reduction and mark FIFO as minimized
          fifo_sizes[i] = oldFifoSize;
          minimizedFifo[i] = true;
        }

        if (fifo_sizes[i] == 1) {
          // Minimum size reached
          minimizedFifo[i] = true;
        }
      }
    }
  }

  return fifo_sizes;
}

int main(int argc, char *argv[]) {

  // Load Kernel and Design
  xsi::Kernel kernel(kernel_libname);
  xsi::Design top(kernel, design_libname, xsim_log_filename, trace_filename);
  using Port = xsi::Port;
  if (trace_filename) {
    // TODO make tracing more finer-grain if possible?
    std::cout << "Tracing enabled for design!!!!!!!!" << std::endl;
    top.trace_all();
  }

  // for(auto&& port : top.ports()) {
  //   std::cout << "Port Name: " << port.name()
  //             << ", Direction: " << (port.dir())
  //             << std::endl;
  // }

  // Simulation Report Statistics
  size_t iters = 0;
  std::deque<size_t> inflightTimestamps;
  inflightTimestamps.push_front(0); // Insert start timestamp of first element

  // Find I/O Streams and initialize their Status
  std::vector<S_AXIS_Control> istreams;
  for (auto &&elem : istream_descs) {
    istreams.emplace_back(top, elem.job_size, elem.job_ticks, elem.name);
  }
  std::vector<M_AXIS_Control> ostreams;
  for (auto &&elem : ostream_descs) {
    ostreams.emplace_back(top, elem.job_size, elem.name);
  }

  // Find Global Control & Run Startup Sequence
  top.run(1000);
  Clock &clk = Clock::initClock(top);
  clearPorts(top);
  reset(top);

  // Start Stream Feed and Capture
  std::cout << "Starting data feed with idle-output timeout of " << max_iters
            << " cycles ...\n"
            << std::endl;

  // Make all Inputs valid & all Outputs ready
  for (auto &&s : istreams) {
    s.valid();
  }
  for (auto &&s : ostreams) {
    s.ready();
  }

  if (istreams.size() > 1 || ostreams.size() > 1) {
    throw std::runtime_error("This simulation is not designed to run with "
                             "multiple input or output streams.");
  }

  std::vector<S_AXI_Control> fifos;


  // For now the number of FIFOs has to be set by hand
  for (size_t i = 0; i < 3; ++i){
    fifos.emplace_back(top, "s_axi_control_0_" + std::to_string(i) + "_");
  }

  auto [start_depth, interval] = determineStartDepth(
      top, clk, istreams, ostreams, inflightTimestamps, iters, fifos);

  auto fifoSizes =
      sizeIteratively(start_depth, interval, clk, top, fifos, istreams,
                      ostreams, inflightTimestamps, iters);

  for (auto &&elem : fifoSizes) {
    std::cout << "FIFO size: " << elem << std::endl;
  }

  return 0;
}
