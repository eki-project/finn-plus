#include <AXI_Control.h>
#include <Clock.h>
#include <Design.h>
#include <Port.h>

#include <bitset>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace xsi;

// Constructor
AXI_Control::AXI_Control(xsi::Design& des, Clock& clock, const std::string& axi_prefix) : prefix(axi_prefix), design(des), clk(clock) {
    // Check if the prefix is valid
    if (prefix.empty()) {
        throw std::invalid_argument("AXI prefix cannot be empty.");
    }

    // Ensure the prefix ends with an underscore
    if (prefix.back() != '_') {
        prefix += "_";
    }
}

// Helper functions for multi-bit signal handling
void AXI_Control::writeAddr(const std::string& signal, uint32_t addr) {
    // Convert addr to binary string
    std::string addr_bin = std::bitset<32>(addr).to_string();

    // Remove leading zeros to get the actual size used in the simulation
    addr_bin.erase(0, addr_bin.find_first_not_of('0'));


    // Get port size
    Port& port = design.getPort(signal);
    auto n_bits = port.width();

    // Ensure the string is the right length
    if (addr_bin.length() < n_bits) {
        addr_bin = std::string(n_bits - addr_bin.length(), '0') + addr_bin;
    } else if (addr_bin.length() > n_bits) {
        addr_bin = addr_bin.substr(addr_bin.length() - n_bits);
    }

    port.set_binstr(addr_bin).write_back();
}

void AXI_Control::writeData(const std::string& signal, uint32_t data) {
    // Similar to write_addr
    std::string data_bin = std::bitset<32>(data).to_string();

    // Get port size
    Port& port = design.getPort(signal);
    auto n_bits = port.width();

    if (data_bin.length() < n_bits) {
        data_bin = std::string(n_bits - data_bin.length(), '0') + data_bin;
    } else if (data_bin.length() > n_bits) {
        data_bin = data_bin.substr(data_bin.length() - n_bits);
    }

    port.set_binstr(data_bin).write_back();
}

void AXI_Control::writeStrb(const std::string& signal, uint32_t strb) {
    // Similar to write_addr
    std::string strb_bin = std::bitset<4>(strb).to_string();

    // Get port size
    Port& port = design.getPort(signal);
    auto n_bits = port.width();

    if (strb_bin.length() < n_bits) {
        strb_bin = std::string(n_bits - strb_bin.length(), '0') + strb_bin;
    } else if (strb_bin.length() > n_bits) {
        strb_bin = strb_bin.substr(strb_bin.length() - n_bits);
    }

    port.set_binstr(strb_bin).write_back();
}

uint32_t AXI_Control::read(const std::string& signal) {
    Port& port = design.getPort(signal);
    return port.read().as_unsigned();
}

void AXI_Control::setBool(const std::string& signal) {
    Port& port = design.getPort(signal);
    port.set(1).write_back();
}

void AXI_Control::clearBool(const std::string& signal) {
    Port& port = design.getPort(signal);
    port.set(0).write_back();
}

bool AXI_Control::chkBool(const std::string& signal) {
    Port& port = design.getPort(signal);
    return port.read().as_bool();
}

void AXI_Control::writeRegister(uint32_t addr, uint32_t data) {
    // Assert BREADY to receive response
    setBool(prefix + "bready");
    // Set address
    writeAddr(prefix + "awaddr", addr);
    // Set data and strobe (full 32-bit word)
    writeData(prefix + "wdata", data);
    writeStrb(prefix + "wstrb", 0xF);  // All bytes enabled

    // Assert AWVALID
    setBool(prefix + "awvalid");

    // Assert WVALID
    setBool(prefix + "wvalid");

    // Wait for AWREADY
    while (!chkBool(prefix + "awready")) {
        clk.toggleClk();
    }

    // Wait for WREADY
    while (!chkBool(prefix + "wready")) {
        clk.toggleClk();
    }

    clk.toggleClk();  // Make sure that for at least one cycle the signals were set

    // Deassert AWVALID and WVALID
    clearBool(prefix + "awvalid");
    clearBool(prefix + "wvalid");


    // Wait for BVALID
    while (!chkBool(prefix + "bvalid")) {
        clk.toggleClk();
    }

    // Check BRESP (optional, could add error handling)
    uint32_t bresp = read(prefix + "bresp");
    if (bresp != 0) {
        std::cerr << "AXI write error: BRESP = " << bresp << std::endl;
    }

    // Deassert BREADY
    clearBool(prefix + "bready");

    clk.toggleClk();
}

uint32_t AXI_Control::readRegister(uint32_t addr) {
    // Assert RREADY to receive data
    setBool(prefix + "rready");
    // Set address
    writeAddr(prefix + "araddr", addr);

    // Assert ARVALID
    setBool(prefix + "arvalid");

    // Wait for ARREADY
    while (!chkBool(prefix + "arready")) {
        clk.toggleClk();
    }

    // Wait for RVALID
    while (!chkBool(prefix + "rvalid")) {
        clk.toggleClk();
    }

    // Deassert ARVALID
    clearBool(prefix + "arvalid");

    // Read data
    uint32_t data = read(prefix + "rdata");

    // Check RRESP (optional, could add error handling)
    uint32_t rresp = read(prefix + "rresp");
    if (rresp != 0) {
        std::cerr << "AXI read error: RRESP = " << rresp << std::endl;
    }

    // Deassert RREADY
    clearBool(prefix + "rready");
    clk.toggleClk();

    return data;
}
