# CMake

**Documentation:** [https://cmake.org/documentation/](https://cmake.org/documentation/)

CMake is a cross-platform build tool that is used to manage software compilation and testing.  From the [CMake web site](https://cmake.org/):

> CMake is an open-source, cross-platform family of tools designed to build, test and package software. CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice.

## Getting Started

On the NREL HPC systems, CMake is available through:

```bash
module load cmake
```

New users are encouraged to refer to the documentation linked above, in particular the [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html).  To build software that includes a `CMakeLists.txt` file, the steps often follow a pattern similar to:

```bash
mkdir build
cd build
# Reference the path to the CMakeLists.txt file:
cmake ..
CC=<c_compiler> CXX=<c++_compiler> make
```

Here the `CC` and `CXX` environment variables are used to explicitly specify the C and C++ compiler that CMake should use.  If not specified, CMake will determine a default compiler to use.
