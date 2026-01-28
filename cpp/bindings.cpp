#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// Example function to verify the C++ bridge is active
void debug_log(const std::string &message) {
    std::cout << "[CPP-LOG] " << message << std::endl;
}

PYBIND11_MODULE(core_cpp, m) {
    m.doc() = "C++ backend for Forensic Weapon Detection";
    m.def("debug_log", &debug_log, "Log a message from C++");
}
