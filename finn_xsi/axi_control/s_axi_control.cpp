#include "s_axi_control.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <bitset>
#include "../xsi_finn.hpp"
#include "../clock/clock.h"

using namespace xsi;

// Constructor
S_AXI_Control::S_AXI_Control(xsi::Design& design, const std::string& axi_prefix) : prefix(axi_prefix), design(design), clk(Clock::initClock(design)) {
    // Check if the prefix is valid
    if (prefix.empty()) {
        throw std::invalid_argument("AXI prefix cannot be empty.");
    }

    // Ensure the prefix ends with an underscore
    if (prefix.back() != '_') {
        prefix += "_";
    }

    // Check if the required ports exist
    if (!design.getPort(prefix + "awaddr") || !design.getPort(prefix + "wdata") ||
        !design.getPort(prefix + "wstrb") || !design.getPort(prefix + "awvalid") ||
        !design.getPort(prefix + "wvalid") || !design.getPort(prefix + "awready") ||
        !design.getPort(prefix + "wready") || !design.getPort(prefix + "bvalid") ||
        !design.getPort(prefix + "bresp") || !design.getPort(prefix + "bready") ||
        !design.getPort(prefix + "araddr") || !design.getPort(prefix + "arvalid") ||
        !design.getPort(prefix + "arready") || !design.getPort(prefix + "rdata") ||
        !design.getPort(prefix + "rvalid") || !design.getPort(prefix + "rready")) {
        // If any required port is missing, throw
        throw std::runtime_error("Required AXI ports not found in design with prefix: " + prefix);
    }
}

// Helper functions for multi-bit signal handling
void S_AXI_Control::write_addr(const std::string& signal, uint32_t addr) {
    // Convert addr to binary string
    std::string addr_bin = std::bitset<32>(addr).to_string();

    // Remove leading zeros to get the actual size used in the simulation
    addr_bin.erase(0, addr_bin.find_first_not_of('0'));


    // Get port size
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    int n_bits = port->width();

    // Ensure the string is the right length
    if (addr_bin.length() < n_bits) {
        addr_bin = std::string(n_bits - addr_bin.length(), '0') + addr_bin;
    } else if (addr_bin.length() > n_bits) {
        addr_bin = addr_bin.substr(addr_bin.length() - n_bits);
    }

    port->set_binstr(addr_bin).write_back();
}

void S_AXI_Control::write_data(const std::string& signal, uint32_t data) {
    // Similar to write_addr
    std::string data_bin = std::bitset<32>(data).to_string();

    // Get port size
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    int n_bits = port->width();

    if (data_bin.length() < n_bits) {
        data_bin = std::string(n_bits - data_bin.length(), '0') + data_bin;
    } else if (data_bin.length() > n_bits) {
        data_bin = data_bin.substr(data_bin.length() - n_bits);
    }

    port->set_binstr(data_bin).write_back();
}

void S_AXI_Control::write_strb(const std::string& signal, uint32_t strb) {
    // Similar to write_addr
    std::string strb_bin = std::bitset<4>(strb).to_string();

    // Get port size
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    int n_bits = port->width();

    if (strb_bin.length() < n_bits) {
        strb_bin = std::string(n_bits - strb_bin.length(), '0') + strb_bin;
    } else if (strb_bin.length() > n_bits) {
        strb_bin = strb_bin.substr(strb_bin.length() - n_bits);
    }

    port->set_binstr(strb_bin).write_back();
}

uint32_t S_AXI_Control::read(const std::string& signal) {
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }

    return port->read().as_unsigned();
}

void S_AXI_Control::set_bool(const std::string& signal) {
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    port->set(1).write_back();
}

void S_AXI_Control::clear_bool(const std::string& signal) {
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    port->set(0).write_back();
}

bool S_AXI_Control::chk_bool(const std::string& signal) {
    Port* port = design.getPort(signal);
    if (!port) {
        throw std::runtime_error("Port " + signal + " not found in design.");
    }
    return port->read().as_bool();
}

void S_AXI_Control::write_register(uint32_t addr, uint32_t data) {
    // Assert BREADY to receive response
    set_bool(prefix + "bready");
    // Set address
    write_addr(prefix + "awaddr", addr);
    // Set data and strobe (full 32-bit word)
    write_data(prefix + "wdata", data);
    write_strb(prefix + "wstrb", 0xF); // All bytes enabled

    // Assert AWVALID
    set_bool(prefix + "awvalid");

    // Assert WVALID
    set_bool(prefix + "wvalid");

    // Wait for AWREADY
    while (!chk_bool(prefix + "awready")) {
        std::cout << "Ready not set\n";
        clk.toggle_clk();
    }

    // Wait for WREADY
    while (!chk_bool(prefix + "wready")) {
        clk.toggle_clk();
    }

    clk.toggle_clk(); //Make sure that for at least one cycle the signals were set

    // Deassert AWVALID and WVALID
    clear_bool(prefix + "awvalid");
    clear_bool(prefix + "wvalid");


    // Wait for BVALID
    while (!chk_bool(prefix + "bvalid")) {
        clk.toggle_clk();
    }

    // Check BRESP (optional, could add error handling)
    uint32_t bresp = read(prefix + "bresp");
    if (bresp != 0) {
        std::cerr << "AXI write error: BRESP = " << bresp << std::endl;
    }

    // Deassert BREADY
    clear_bool(prefix + "bready");

    clk.toggle_clk();
}

uint32_t S_AXI_Control::read_register(uint32_t addr) {
    // Assert RREADY to receive data
    set_bool(prefix + "rready");
    // Set address
    write_addr(prefix + "araddr", addr);

    // Assert ARVALID
    set_bool(prefix + "arvalid");

    // Wait for ARREADY
    while (!chk_bool(prefix + "arready")) {
        clk.toggle_clk();
    }

    // Wait for RVALID
    while (!chk_bool(prefix + "rvalid")) {
        clk.toggle_clk();
    }

    // Deassert ARVALID
    clear_bool(prefix + "arvalid");

    // Read data
    uint32_t data = read(prefix + "rdata");

    // Check RRESP (optional, could add error handling)
    uint32_t rresp = read(prefix + "rresp");
    if (rresp != 0) {
        std::cerr << "AXI read error: RRESP = " << rresp << std::endl;
    }

    // Deassert RREADY
    clear_bool(prefix + "rready");
    clk.toggle_clk();

    return data;
}
