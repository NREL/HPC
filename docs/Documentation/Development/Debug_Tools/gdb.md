# GDB (GNU Debugger)

**Documentation:** [GDB](https://www.sourceware.org/gdb/)

*GDB is GNU's command line interface debugging tool.*

## Getting started

GDB is available on NREL machines and supports a number of languages, including C, C++, and Fortran. 

When using GDB, make sure the program you are attempting to debug has been compiled with the `-g` debug flag and with the `-O0` optimization flag to achieve the best results.

Run GDB with the following command: `gdb --args my_executable arg1 arg 2 arg3`
This will launch gdb running `my_executable`, and passes arguments `arg1`, `arg2`, and `arg3` to `my_executable`.

For links to in-depth tutorials and walkthroughs of GDB features, please see [Resources](#resources).

## Availability

| Eagle | Swift | Vermilion |
|:-------:|:-----:|:-----------|
| gdb/7.6.1\*| gdb/8.2\*  |gdb/12.1, gdb/8.2\*  |

\* Located in `/usr/bin`. Do not need to use `module load`.

## Resources

* [Introduction to GDB](https://www.cs.umd.edu/~srhuang/teaching/cmsc212/gdb-tutorial-handout.pdf)

* [Sample GDB session](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Sample-Session.html#Sample-Session)

* ["Print statement"-style debugging with GDB](https://developers.redhat.com/articles/2021/10/05/printf-style-debugging-using-gdb-part-1#)
