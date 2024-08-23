# C++

*"C++ is a general-purpose programming language providing a direct and efficient model of hardware combined with facilities for defining lightweight abstractions."*
  - Bjarne Stroustrup, "The C++ Programming Language, Fourth Edition"

## Getting Started

This section illustrates the process to compile and run a basic C++ program on the HPC systems.

### Hello World

Begin by creating a source file named `hello.cpp` with the following contents:

```c++
#include <iostream>

int main(void) {
  std::cout << "Hello, World!\n";
  return 0;
}
```

Next, we must select the compiler to use for compiling our program.  We can choose among GNU, Intel, and Cray compilers, depending on the system that we are using (see [Compilers and Toolchains](#compilers-and-toolchains)).  To see available modules and versions, use `module avail`.  For this example, we will use the `g++` compiler, which is part of GNU's `gcc` package.  We will load the default version of the compiler, which in this case is gcc 10.1:

```
$ module load gcc
$ module list
Currently Loaded Modules:
  1) gcc/10.1.0
$ gcc --version | head -1
gcc (Spack GCC) 10.1.0
```

With the `gcc` package, the C++ compiler is provided by the `g++` command.  To compile the program, run:

```
$ g++ hello.cpp -o hello
```

This creates an executable named `hello`.  Now run the program and observe the output:

```
$ ./hello
Hello, World!
```

## Compilers and Toolchains

The following is a summary of available compilers and toolchains.  User are encouraged to run `module avail` to check for the most up-to-date information on a particular system.

| Toolchain | C++ Compiler | Module                   | Systems                   |
|-----------|--------------|--------------------------|---------------------------|
| gcc       | `g++`        | `gcc`                    | All                       |
| Intel     | `icpc`       | `intel-oneapi-compilers` | Swift, Vermilion, Kestrel |
| Cray      | `CC`         | `PrgEnv-cray`            | Kestrel                   |

Note that Kestrel also provides the `PrgEnv-intel` and `PrgEnv-gnu` modules, which combine the Intel or gcc compilers together with Cray MPICH.  Please refer to [Kestrel Programming Environments Overview](../../Systems/Kestrel/Environments/index.md) for details about the programming environments available on Kestrel.

For information specific to compiling MPI applications, refer to [MPI](../Programming_Models/mpi.md).
