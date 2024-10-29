```
Get an interactive session on a Kestrel GPU node

sbatch script

Any directory that has a "doit" file has a
working example.

See the file about.md for a description of
the compile and run options for each example.

```


```
.
├── README.md
├── about.md
├── cleanup
├── cuda
│   ├── cray
│   │   ├── doit
│   │   └── stream.cu -> ../src/stream.cu
│   ├── gccalso
│   │   ├── cuda.cu -> ../src/cuda.cu
│   │   ├── doit
│   │   ├── doswift
│   │   ├── extras.h -> ../src/extras.h
│   │   └── normal.c -> ../src/normal.c
│   ├── nvidia
│   │   ├── doit
│   │   ├── doswift
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
│   │   ├── doit
│   │   └── doswift
│   └── fft
│       ├── 3d_mgpu_c2c_example.cpp
│       ├── cufft_utils.h
│       ├── doit
│       ├── doswift
│       └── fftw3d.c
├── makeit
├── mpi
│   ├── cudaaware
│   │   ├── all2all.cu -> src/all2all.cu
│   │   ├── check.cu -> src/check.cu
│   │   ├── cpumod.c -> src/cpumod.c
│   │   ├── doit
│   │   ├── doswift
│   │   ├── gpumod.cu -> src/gpumod.cu
│   │   ├── ping_pong_cuda_aware.cu -> src/ping_pong_cuda_aware.cu
│   │   └── src
│   │       ├── all2all.cu
│   │       ├── call2all.c
│   │       ├── check.cu
│   │       ├── cpumod.c
│   │       ├── gpumod.cu
│   │       ├── hold.c
│   │       ├── ping_pong_cuda_aware.cu
│   │       ├── qtf
│   │       │   ├── normal
│   │       │   ├── nvhpc.qtf
│   │       │   ├── res.qtf
│   │       │   ├── simple.qtf
│   │       │   └── slurm.qtf
│   │       └── testit
│   ├── normal
│   │   ├── cray
│   │   │   ├── doit
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   └── hellof.f90 -> ../src/hellof.f90
│   │   ├── intel+abi
│   │   │   ├── docpu
│   │   │   ├── doit
│   │   │   ├── doswift
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   ├── hellof.f90 -> ../src/hellof.f90
│   │   │   └── oncpu
│   │   ├── nvidia
│   │   │   ├── nrelopenmpi
│   │   │   │   ├── doit
│   │   │   │   ├── doswift
│   │   │   │   ├── helloc.c -> ../../src/helloc.c
│   │   │   │   └── hellof.f90 -> ../../src/hellof.f90
│   │   │   └── nvidiaopenmpi
│   │   │       ├── doit
│   │   │       ├── doswift
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
│   │   │   │   ├── doit
│   │   │   │   └── doswift
│   │   │   └── nvidiaopenmpi
│   │   │       ├── acc_c3.c -> ../../src/acc_c3.c
│   │   │       ├── doit
│   │   │       └── doswift
│   │   └── src
│   │       └── acc_c3.c
│   └── withcuda
│       ├── cray
│       │   ├── doit
│       │   ├── mstream.cu -> ../src/mstream.cu
│       │   └── ping_pong_cuda_staged.cu -> ../src/ping_pong_cuda_staged.cu
│       ├── nvidia
│       │   ├── nrelopenmpi
│       │   │   ├── doit
│       │   │   ├── doswift
│       │   │   ├── mstream.cu -> ../../src/mstream.cu
│       │   │   └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       │   └── nvidiaopenmpi
│       │       ├── doit
│       │       ├── doswift
│       │       ├── mstream.cu -> ../../src/mstream.cu
│       │       └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       └── src
│           ├── mstream.cu
│           └── ping_pong_cuda_staged.cu
├── onnodes
├── openacc
│   ├── cray
│   │   ├── doit
│   │   └── nbodyacc2.c -> ../src/nbodyacc2.c
│   ├── nvidia
│   │   ├── doit
│   │   ├── doswift
│   │   └── nbodyacc2.c -> ../src/nbodyacc2.c
│   └── src
│       └── nbodyacc2.c
├── output
│   ├── kestrel.env
│   ├── kestrel.info
│   ├── kestrel.out
│   ├── quick.out
│   ├── swift.info
│   └── swift.out
├── quick
├── runall
├── script
├── slides.pdf
├── swift
├── tests
└── whack.sh

36 directories, 109 files
```
