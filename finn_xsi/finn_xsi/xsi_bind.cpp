/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Python binding for FINN XSI++.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

#include <Design.h>
#include <Kernel.h>
#include <Port.h>
#include <pybind11/pybind11.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>


namespace py = pybind11;
using namespace xsi;

namespace {
    std::mutex use_mutex;
    // Keeps the Kernel that backs a Design alive for as long as the Design exists and
    // records which design library it was opened from, so that an attempt to open a
    // second design can be reported with a useful message.
    struct DesignUse {
        std::shared_ptr<Kernel> kernel;
        std::string design_lib;
    };
    std::map<Design const*, DesignUse> use_map;

    struct DesignDeleter : public std::default_delete<Design> {
        void operator()(Design* d) const {
            std::default_delete<Design>::operator()(d);
            std::lock_guard<std::mutex> lock(use_mutex);
            use_map.erase(d);
        }
    };

    // The XSI simulator kernel keeps process-global state, and all design libraries
    // export the same symbols. Having two designs open at the same time therefore
    // corrupts the simulation and typically ends in a hang or a segfault deep inside
    // xsi_open. Refuse it with a comprehensible error instead.
    void assert_no_open_design(std::string const& design_lib) {
        std::lock_guard<std::mutex> lock(use_mutex);
        for (auto const& [design, use] : use_map) {
            if (design->is_open()) {
                throw std::runtime_error("Cannot open XSI design '" + design_lib + "': design '" + use.design_lib +
                                         "' is still open in this process. Only one XSI design can be open at a time; close the previous "
                                         "simulation before starting a new one.");
            }
        }
    }
}  // namespace

PYBIND11_MODULE(xsi, m) {
    py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel").def(py::init<std::string const&>());

    py::class_<Design, std::unique_ptr<Design, DesignDeleter>>(m, "Design")
        .def(py::init([](std::shared_ptr<Kernel> const& kernel, std::string const& design_lib, char const* const log_file, char const* const wdb_file) {
            assert_no_open_design(design_lib);
            std::unique_ptr<Design, DesignDeleter> d{new Design(*kernel, design_lib, log_file, wdb_file)};
            std::lock_guard<std::mutex> lock(use_mutex);
            use_map.emplace(d.get(), DesignUse{kernel, design_lib});
            return d;
        }))
        .def("close", &Design::close)
        .def("is_open", &Design::is_open)
        .def("trace_all", &Design::trace_all)
        .def("run", &Design::run)
        .def("restart", &Design::restart)
        .def("get_status", &Design::get_status)
        .def("get_error_info", &Design::get_error_info)
        .def("num_ports", &Design::num_ports)
        .def("getPort", static_cast<Port& (Design::*) (std::string const&)>(&Design::getPort), py::return_value_policy::reference_internal)
        .def(
            "ports",
            [](Design& d) {
                auto const e = d.ports();
                return py::make_iterator(e.begin(), e.end(), py::return_value_policy::reference_internal);
            },
            py::keep_alive<0, 1>());

    py::class_<Port, std::unique_ptr<Port, py::nodelete>>(m, "Port")
        .def("name", &Port::name)
        .def("dir", &Port::dir)
        .def("width", &Port::width)
        .def("isInput", &Port::isInput)
        .def("isOutput", &Port::isOutput)
        .def("isInout", &Port::isInout)
        .def("read", &Port::read, py::return_value_policy::reference_internal)
        .def("write_back", &Port::write_back)
        .def("hasUnknown", &Port::hasUnknown)
        .def("isZero", &Port::isZero)
        .def("as_bool", &Port::as_bool)
        .def("as_unsigned", &Port::as_unsigned)
        .def("as_binstr", &Port::as_binstr)
        .def("as_hexstr", &Port::as_hexstr)
        .def("clear", &Port::clear, py::return_value_policy::reference_internal)
        .def("set", &Port::set, py::return_value_policy::reference_internal)
        .def("set_binstr", &Port::set_binstr, py::return_value_policy::reference_internal)
        .def("set_hexstr", &Port::set_hexstr, py::return_value_policy::reference_internal);
}
