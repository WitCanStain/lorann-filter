import glob
import os
import pathlib
import re
import shutil
import sys

import cffi
import invoke

@invoke.task
def clean(c):
    """Remove any built objects"""
    for file_pattern in (
        "*.o",
        "*.so",
        "*.obj",
        "*.dll",
        "*.exp",
        "*.lib",
        "*.pyd",
        "cffi_example*",  # Is this a dir?
        "cython_wrapper.cpp",
    ):
        for file in glob.glob(file_pattern):
            os.remove(file)
    for dir_pattern in "Release":
        for dir in glob.glob(dir_pattern):
            shutil.rmtree(dir)

@invoke.task()
def build_cppmult(c):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC filter.cpp "
        "-o libfilter.so "
    )
    print("* Complete")

@invoke.task()
def build_cppmult_test(c):
    """Build the shared library for the sample C++ code"""
    print_banner("Building C++ Library")
    invoke.run(
        "g++ -std=c++17 -O3 -march=native -shared -fopenmp -fPIC -DNDEBUG -I../../lorann -flax-vector-conversions filter.cpp -o filter -lgomp"
        "-o libfilter.so "
    )
    print("* Complete")


@invoke.task()
def test_ctypes_cpp(c):
    """Run the script to test filter"""
    print_banner("Testing ctypes Module for C++")
    # pty and python3 didn't work for me (win).
    if on_win:
        invoke.run("python experimenter.py")
    else:
        invoke.run("python3 experimenter.py", pty=True)
        