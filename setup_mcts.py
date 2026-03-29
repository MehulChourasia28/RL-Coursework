"""
Build the C++ MCTS extension in-place.

Usage:
    python setup_mcts.py build_ext --inplace
"""
from setuptools import setup, Extension
import pybind11

ext = Extension(
    "Mehuls_agent.mcts_cpp",
    sources=["Mehuls_agent/mcts_cpp.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=[
        "-O3", "-std=c++17", "-march=native",
        "-fvisibility=hidden",   # required by pybind11
        "-Wall", "-Wextra", "-Wpedantic",
    ],
    language="c++",
)

setup(
    name="mcts_cpp",
    ext_modules=[ext],
)
