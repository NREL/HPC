To run get an interactive session on a Kestrel RHEL9 GPU login node.
Then submit the script doall.  

sbatch --account=MY_ACCOUNT_NAME doall

This will run all of the examples.  It should take about 20 minutes. 
The file output-*.out will contain the combined output from each program.  
The file infor-*.out will contain stderr from the builds.  There will be
a few warnings but there should not be any errors.

Any directory that has a "doit" file has a working example and will be run.  

See the file details.md for more information and a description of
the compile and run options for each example.

These files have been modified to work on Kestrel's RHEL9
operating system.  The "*nlropenmpi*" are not supported
under RHEL9 the "doit" file in these directories have been
moved to "dont"

```
├── README.md
├── about.md
├── bw.png
├── cleanup
├── cuda
│   ├── cray
│   │   ├── doit
│   │   └── stream.cu -> ../src/stream.cu
│   ├── gccalso
│   │   ├── cuda.cu -> ../src/cuda.cu
│   │   ├── doit
│   │   ├── extras.h -> ../src/extras.h
│   │   └── normal.c -> ../src/normal.c
│   ├── nvidia
│   │   ├── doit
│   │   └── stream.cu -> ../src/stream.cu
│   └── src
│       ├── cuda.cu
│       ├── extras.h
│       ├── normal.c
│       ├── qnd.cu
│       └── stream.cu
├── cudalib
│   ├── factor
│   │   ├── cpu.C
│   │   ├── cusolver_getrf_example.cu
│   │   ├── cusolver_utils.h
│   │   └── doit
│   └── fft
│       ├── 3d_mgpu_c2c_example.cpp
│       ├── cufft_utils.h
│       ├── doit
│       └── fftw3d.c
├── doall
├── each
├── mpi
│   ├── cudaaware
│   │   ├── all2all.cu -> src/all2all.cu
│   │   ├── check.cu -> src/check.cu
│   │   ├── cpumod.c -> src/cpumod.c
│   │   ├── doit
│   │   ├── gpumod.cu -> src/gpumod.cu
│   │   ├── ping_pong_cuda_aware.cu -> src/ping_pong_cuda_aware.cu
│   │   └── src
│   │       ├── all2all.cu
│   │       ├── call2all.c
│   │       ├── check.cu
│   │       ├── cpumod.c
│   │       ├── gpumod.cu
│   │       ├── hold.c
│   │       ├── ping_pong_cuda_aware.cu
│   │       ├── qtf
│   │       │   ├── normal
│   │       │   ├── nvhpc.qtf
│   │       │   ├── res.qtf
│   │       │   ├── simple.qtf
│   │       │   └── slurm.qtf
│   │       └── testit
│   ├── normal
│   │   ├── cray
│   │   │   ├── doit
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   └── hellof.f90 -> ../src/hellof.f90
│   │   ├── intel+abi
│   │   │   ├── docpu
│   │   │   ├── doit
│   │   │   ├── helloc.c -> ../src/helloc.c
│   │   │   ├── hellof.f90 -> ../src/hellof.f90
│   │   │   └── oncpu
│   │   ├── nvidia
│   │   │   ├── nlropenmpi
│   │   │   │   ├── dont
│   │   │   │   ├── helloc.c -> ../../src/helloc.c
│   │   │   │   └── hellof.f90 -> ../../src/hellof.f90
│   │   │   └── nvidiaopenmpi
│   │   │       ├── doit
│   │   │       ├── helloc.c -> ../../src/helloc.c
│   │   │       └── hellof.f90 -> ../../src/hellof.f90
│   │   └── src
│   │       ├── helloc.c
│   │       └── hellof.f90
│   ├── openacc
│   │   ├── cray
│   │   │   ├── acc_c3.c -> ../src/acc_c3.c
│   │   │   └── doit
│   │   ├── nvidia
│   │   │   ├── nlropenmpi
│   │   │   │   ├── acc_c3.c -> ../../src/acc_c3.c
│   │   │   │   └── dont
│   │   │   └── nvidiaopenmpi
│   │   │       ├── acc_c3.c -> ../../src/acc_c3.c
│   │   │       └── doit
│   │   └── src
│   │       └── acc_c3.c
│   └── withcuda
│       ├── cray
│       │   ├── doit
│       │   ├── mstream.cu -> ../src/mstream.cu
│       │   └── ping_pong_cuda_staged.cu -> ../src/ping_pong_cuda_staged.cu
│       ├── nvidia
│       │   ├── nlropenmpi
│       │   │   ├── dont
│       │   │   ├── mstream.cu -> ../../src/mstream.cu
│       │   │   └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       │   └── nvidiaopenmpi
│       │       ├── doit
│       │       ├── mstream.cu -> ../../src/mstream.cu
│       │       └── ping_pong_cuda_staged.cu -> ../../src/ping_pong_cuda_staged.cu
│       ├── openmpi
│       │   ├── doit
│       │   ├── mstream.cu -> ../src/mstream.cu
│       │   └── ping_pong_cuda_staged.cu -> ../src/ping_pong_cuda_staged.cu
│       └── src
│           ├── mstream.cu
│           └── ping_pong_cuda_staged.cu
├── onnodes
├── openacc
│   ├── cray
│   │   ├── doit
│   │   └── nbodyacc2.c -> ../src/nbodyacc2.c
│   ├── nvidia
│   │   ├── doit
│   │   └── nbodyacc2.c -> ../src/nbodyacc2.c
│   └── src
│       └── nbodyacc2.c
├── picenv
├── picenv.md
├── picmod -> picenv
├── quick
├── results.tgz
├── runall
├── slides.pdf
├── tarup
├── tests
└── update.py
```

