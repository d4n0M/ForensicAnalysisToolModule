from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "inference.core_cpp",
        ["cpp/bindings.cpp"],
        cxx_std=17
    ),
]

setup(
    name="forensic_weapon_detection",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
