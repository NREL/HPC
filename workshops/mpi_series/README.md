# Message Passing Interface Library (MPI) 4 Part Series

## Session 1: Introduction

The first session will introduce MPI. We will give a background, show some sources of Documentation. We will show the classic "Hello world" program in MPI running on multiple processors. We will discuss Basic communications and show a simple send and receive program where messages are passed between processors. We will be running examples in the official languages supported by MPI, C and Fortran. Also, we will briefly discuss support for Python, R, and Java.  Source code and scripts will be provided that can be run on Eagle.

## Session 2: Expansion to Higher-level Calls

The second session will expand on the first, showing many of the higher-level MPI calls commonly used to write parallel programs. Examples will be provided that can run on Eagle. We will look at: using the various predefined data types, broadcast, wildcards, asynchronous communications, and using probes and status information to control flow.

## Session 3: Additional Collective Operations

In the third MPI session, we will look at additional collective operations including scatter, gather, and reductions. We’ll also show examples of the "variable" versions of these calls where the amount of information shared is processor-dependent. We’ll look at creating derived data types and managing subsets of processes using communicators.

## Session 4: Finite Difference Model

In the fourth session, we will introduce a finite difference model that will demonstrate what a computational scientist needs to do to take advantage of computers using MPI. The model we are using is a two-dimensional finite-difference code. After discussing the serial code, we will show the modifications necessary to turn it into a parallel program using MPI. We will look at domain decomposition, initialization, data distribution, message passing, reduction operations, and multiple methods for data output. We will also look at the performance of the application on various numbers of processors to illustrate Amdahl's parallel program scaling law.

---

**[Code Examples](https://github.com/timkphd/examples/tree/master/mpi)**
