/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Driver harness demo running a FINN IP core.
 * @author	Yaman Umuroğlu <yaman.umuroglu@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @author  Linus Jungemann <linus.jungemann@uni-paderborn.de>
 ***************************************************************************/

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>
#include <tuple>
#include <functional>
#include <cstring>

#include <AXI_Control.h>
#include <AXIS_Control.h>
#include <Clock.h>
#include <Kernel.h>
#include <Design.h>
#include <Port.h>

#include "rtlsim_config.hpp"

void clearPorts(xsi::Design &top) {
  // Clear all input ports
  for (xsi::Port &p : top.ports()) {
    if (p.isInput()) {
      p.clear().write_back();
    }
  }
}

void reset(xsi::Design &top) {
  xsi::Port& rst_n = top.getPort("ap_rst_n");
  Clock &clk = Clock::initClock(top);
  if (!rst_n) {
    std::cerr << "No reset port found in design." << std::endl;
    return;
  }
  // Reset all Inputs, Wait for Reset Period
  rst_n->set(0).write_back();
  if (rst_n) {
    for (unsigned i = 0; i < 16; i++) {
      clk.toggle_clk();
    }
    rst_n->set(1).write_back();
  }
}

// Helper function to process input streams
void processInputStream(S_AXIS_Control &stream, size_t iters,
                        size_t &inputStreamsRemaining, size_t maxInferences,
                        std::vector<std::reference_wrapper<xsi::Port>> &deferredWrites) {
  const bool isValid = stream.is_valid();

  // Skip if valid but not ready
  if (isValid && !stream.is_ready()) {
    return;
  }

  // Track successful transactions
  if (isValid) {
    stream.job_txns++;
    if (++stream.total_txns == stream.job_size * maxInferences) {
      inputStreamsRemaining--;
    }
  }

  const bool shouldProcessJob =
      (stream.job_txns < stream.job_size) || (iters >= stream.await_iter);
  const bool hasMoreTransactions =
      stream.total_txns < stream.job_size * maxInferences;

  // Handle throttling and job completion
  if (shouldProcessJob && hasMoreTransactions) {
    if (!isValid) {
      deferredWrites.emplace_back(stream.set_valid(true));
    }

    // Reset job when completed
    if (stream.job_txns == stream.job_size) {
      stream.job_txns = 0;
      stream.await_iter = iters + stream.job_ticks;
    }
  } else if (isValid) {
    deferredWrites.emplace_back(stream.set_valid(false));
  }
}

// Helper function to process output streams
bool processOutputStream(M_AXIS_Control &stream, size_t iters,
                         size_t &outputStreamsRemaining, size_t &muteOutputs,
                         size_t maxInferences,
                         std::vector<std::reference_wrapper<xsi::Port>> &deferredWrites) {
  if (!stream.is_ready() || !stream.is_valid()) {
    return false;
  }

  const size_t totalTransactions = ++stream.total_txns;

  // Track first completion
  if (totalTransactions == stream.job_size) {
    stream.first_complete = iters;
    muteOutputs--;
  }

  // Track job completion and intervals
  if (++stream.job_txns == stream.job_size) {
    stream.interval = iters - stream.last_complete;
    stream.last_complete = iters;
    stream.job_txns = 0;
  }

  // Handle final completion or spurious outputs
  if (totalTransactions >= stream.job_size * maxInferences) {
    if (totalTransactions == stream.job_size * maxInferences) {
      outputStreamsRemaining--;
    } else {
      deferredWrites.emplace_back(stream.set_ready(false));
    }
  }

  return true;
}

int main(int argc, char *argv[]) {

  // Load Kernel and Design
  xsi::Kernel kernel(kernel_libname);
  xsi::Design top(kernel, design_libname, xsim_log_filename, trace_filename);
  using Port = xsi::Port;
  if (trace_filename) {
    // TODO make tracing more finer-grain if possible?
    top.trace_all();
  }

  // Ultimate Simulation Summary
  std::string synopsis;

  // Simulation Report Statistics
  size_t iters = 0;
  size_t timeout = 0;
  size_t itodo = istream_descs.size();
  size_t otodo = ostream_descs.size();
  size_t omute = ostream_descs.size();

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

  // Enter Simulation Loop and track Progress
  auto const begin = std::chrono::steady_clock::now();
  std::vector<std::reference_wrapper<Port>> to_write;
  while (true) {

    //-------------------------------------------------------------------
    // Clock down - then read signal updates from design
    clk.cycle(0);

	//-------------------------------------------------------------------
    // Process input streams

    for (auto &stream : istreams) {
      processInputStream(stream, iters, itodo, n_inferences, to_write);
    }

    //-------------------------------------------------------------------
    // Process output streams

    bool hasActiveOutput = false;
    for (auto &stream : ostreams) {
      if (processOutputStream(stream, iters, otodo, omute, n_inferences, to_write)) {
        hasActiveOutput = true;
      }
    }
    timeout = hasActiveOutput ? 0 : timeout + 1;

    //-------------------------------------------------------------------
    // Clock up - then write signal updates back to design
    clk.cycle(1);

    // Write back Ports with registered updates
    for (Port &p : to_write)
      p.write_back();
    to_write.clear();

    // Show a progress message once in a while
    if (++iters % 10000 == 0) {
      std::cout << '@' << iters << " ticks / "
                << std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - begin)
                       .count()
                << "s:";
      for (auto const &s : istreams) {
        std::cout << '\t' << s.name << '='
                  << ((100 * s.total_txns) / (n_inferences * s.job_size))
                  << '%';
      }
      for (auto const &s : ostreams) {
        std::cout << '\t' << s.name << '='
                  << ((100 * s.total_txns) / (n_inferences * s.job_size))
                  << '%';
      }
      std::cout << "\tMute Outputs: " << omute << std::endl;
    }

    // Check for exit
    if ((timeout > max_iters) || (!itodo && !otodo))
      break;
  }

  size_t total_in_txns = 0;
  for (auto const &s : istreams)
    total_in_txns += s.total_txns;

  size_t total_out_txns = 0;
  size_t firstout_latency = 0;
  size_t max_interval = 0;
  for (auto const &s : ostreams) {
    total_out_txns += s.total_txns;
    firstout_latency = std::max(firstout_latency, s.first_complete);
    max_interval = std::max(max_interval, s.interval);
  }

  std::ostringstream bld;
  bld << "N_IN_TXNS\t" << total_in_txns
      << "\n"
         "N_OUT_TXNS\t"
      << total_out_txns
      << "\n"
         "cycles\t"
      << iters
      << "\n"
         "N\t"
      << n_inferences
      << "\n"
         "latency_cycles\t"
      << firstout_latency
      << "\n"
         "interval_cycles\t"
      << max_interval
      << "\n"
         "TIMEOUT\t"
      << (timeout > max_iters ? "1" : "0")
      << "\n"
         "UNFINISHED_INS\t"
      << itodo
      << "\n"
         "UNFINISHED_OUTS\t"
      << otodo
      << "\n"
         "RUNTIME_S\t"
      << std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::steady_clock::now() - begin)
             .count();
  synopsis = bld.str();

  // Dump Simulation Statistics to stdout and results.txt
  std::cout << '\n' << synopsis << std::endl;

  // Log error info to file
  std::ofstream error_file("fifosim.err", std::ios::out | std::ios::trunc);
  error_file << top.get_error_info();

  // Synopsis and `max_count` readings to results file
  std::ofstream results_file("results.txt", std::ios::out | std::ios::trunc);
  results_file << synopsis << std::endl;
  for (Port &p : top.ports()) {
    if (p.isOutput()) {
      char const *const name = p.name();
      if (std::strncmp(name, "maxcount", 8) == 0) {
        p.read();
        results_file << name << '\t' << p.as_unsigned() << std::endl;
      }
    }
  }

  return 0;
}
