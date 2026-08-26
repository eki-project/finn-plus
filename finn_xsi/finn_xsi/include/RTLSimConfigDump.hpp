#ifndef RTLSIMCONFIGDUMP
#define RTLSIMCONFIGDUMP

#include <SocketServer.h>
#include <rtlsim_config.hpp>
#include <string>

// Dump this binary's compiled-in RTLSimConfig (baked in at build time from this node's
// rtlsim_config.hpp) as JSON. Lets you inspect on disk which node a given simulation
// binary was ACTUALLY built for, independent of what the rtlsim_config.hpp file sitting
// next to it currently says -- useful when a build-time bug causes those two to diverge
// (e.g. a stale/cross-contaminated compile).
inline json dump_rtlsim_config() {
    json cfg;
    cfg["node_index"] = RTLSimConfig::NodeIndex;
    cfg["total_nodes"] = RTLSimConfig::TotalNodes;
    cfg["is_input_node"] = RTLSimConfig::IsInputNode;
    cfg["is_output_node"] = RTLSimConfig::IsOutputNode;
    cfg["sim_comm_mode"] = std::string(RTLSimConfig::SimCommMode);
    cfg["kernel_libname"] = std::string(RTLSimConfig::kernel_libname);
    cfg["design_libname"] = std::string(RTLSimConfig::design_libname);
    cfg["max_iters"] = RTLSimConfig::max_iters;
    cfg["precise_timeout"] = RTLSimConfig::preciseTimeout;
    cfg["trace_filename"] = RTLSimConfig::trace_filename.value_or("");
    cfg["xsim_log_filename"] = RTLSimConfig::xsim_log_filename;

    auto to_str_array = [](const auto& arr) {
        json j = json::array();
        for (const auto& v : arr) j.push_back(std::string(v));
        return j;
    };
    auto to_int_array = [](const auto& arr) {
        json j = json::array();
        for (const auto& v : arr) j.push_back(v);
        return j;
    };
    auto to_stream_desc_array = [](const auto& arr) {
        json j = json::array();
        for (const auto& d : arr) {
            json entry;
            entry["name"] = std::string(d.name);
            entry["job_size"] = d.job_size;
            j.push_back(entry);
        }
        return j;
    };

    cfg["input_interface_names"] = to_str_array(RTLSimConfig::inputInterfaceNames);
    cfg["output_interface_names"] = to_str_array(RTLSimConfig::outputInterfaceNames);
    cfg["input_channel_types"] = to_str_array(RTLSimConfig::inputChannelTypes);
    cfg["output_channel_types"] = to_str_array(RTLSimConfig::outputChannelTypes);
    cfg["input_peer_ranks"] = to_int_array(RTLSimConfig::inputPeerRanks);
    cfg["output_peer_ranks"] = to_int_array(RTLSimConfig::outputPeerRanks);
    cfg["istream_descs"] = to_stream_desc_array(RTLSimConfig::istream_descs);
    cfg["ostream_descs"] = to_stream_desc_array(RTLSimConfig::ostream_descs);

    return cfg;
}

#endif /* RTLSIMCONFIGDUMP */
