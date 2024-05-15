```
Get an interactive session on 2 h100 nodes.

Source the file runall to run all examples.

Any directory that has a "doit" file has a
working example.

See the file about.md for a description of
the compile and run options for each example.

.
├── cuda
│   ├── cray
│   │   ├── doit
│   │   └── stream.cu -> ../src/stream.cu
│   ├── gccalso
│   │   ├── cuda.cu -> ../src/cuda.cu
│   │   ├── doit
│   │   ├── extras.h -> ../src/extras.h
│   │   └── normal.c -> ../src/normal.c
│   ├── nvidia
│   │   ├── doit
│   │   └── stream.cu -> ../src/stream.cu
│   └── src
│       ├── cuda.cu
│       ├── extras.h
│       ├── normal.c
│       └── stream.cu
├── cudalib
│   ├── factor
│   │   ├── cpu.C
│   │   ├── cusolver_getrf_example.cu
│   │   ├── cusolver_utils.h
│   │   └── doit
│   └── fft
│       ├── 3d_mgpu_c2c_example.cpp
│       ├── 3d_mgpu_c2c_example.o
│       ├── cufft_utils.h
│       └── doit
├── mpi
│   ├── cudaaware
│   │   ├── src
│   │   │   └── ping_pong_cuda_aware.cu
│   │   ├── doit
│   │   └── ping_pong_cuda_aware.cu -> src/ping_pong_cuda_aware.cu
│   ├── normal
│   │   ├── cray
│   │   │   ├── doit
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   └── hellof.f90 -> ../src/hellof.f90
│   │   ├── intel+abi
│   │   │   ├── doit
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   └── hellof.f90 -> ../src/hellof.f90
│   │   ├── nvidia
│   │   │   ├── nrelopenmpi
│   │   │   │   ├── doit
│   │   │   │   ├── helloc.c -> ../../src/helloc.c
│   │   │   │   └── hellof.f90 -> ../../src/hellof.f90
│   │   │   └── nvidiaopenmpi
│   │   │       ├── doit
│   │   │       ├── helloc.c -> ../../src/helloc.c
│   │   │       └── hellof.f90 -> ../../src/hellof.f90
│   │   └── src
│   │       ├── helloc.c
│   │       └── hellof.f90
│   ├── openacc
│   │   ├── cray
│   │   │   ├── acc_c3.c -> ../src/acc_c3.c
│   │   │   └── doit
│   │   ├── nvidia
│   │   │   ├── nrelopenmpi
│   │   │   │   ├── acc_c3.c -> ../../src/acc_c3.c
│   │   │   │   └── doit
│   │   │   └── nvidiaopenmpi
│   │   │       ├── acc_c3.c -> ../../src/acc_c3.c
│   │   │       └── doit
│   │   └── src
│   │       └── acc_c3.c
│   └── withcuda
│       ├── cray
│       │   ├── doit
│       │   └── ping_pong_cuda_staged.cu -> ../src/ping_pong_cuda_staged.cu
│       ├── nvidia
│       │   ├── nrelopenmpi
│       │   │   ├── doit
│       │   │   └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       │   └── nvidiaopenmpi
│       │       ├── doit
│       │       └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       └── src
│           └── ping_pong_cuda_staged.cu
├── openacc
    ├── cray
    │   ├── doit
    │   └── nbodyacc2.c -> ../src/nbodyacc2.c
    ├── nvidia
    │   ├── doit
    │   └── nbodyacc2.c -> ../src/nbodyacc2.c
    └── src
        └── nbodyacc2.c

```

