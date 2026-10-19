# Environments tutorial

In this tutorial, we will walk through how to build and run a basic MPI code using three different toolchains/software stacks on Kestrel.

We summarize these toolchains in the below table:

| PrgEnv-* | Compiler | MPI |
|----------|----------|--------------------|
|cray| cray cce| Cray MPICH |
|intel| intel | Cray MPICH | 
|n/a| intel | Intel MPI | 

## Introduction

Kestrel is a Cray machine whose nodes are connected by "Cray Slingshot" network (contrast this to Eagle, which uses infiniband). We've found that packages that make use of Cray tools like Cray MPICH perform faster than when the same package is built and run without Cray tools (e.g. compiling and running with Intel MPI), in part because these Cray tools are optimized to work well with Cray Slingshot.

Most of us coming from Eagle are probably used to running our codes with Intel MPI or Open MPI, but not Cray MPICH.

Using the cray-designed programming environments ("Prg-Envs") requires using special Cray compiler wrappers `cc` and `ftn`. These wrappers replace the MPI compiler wrappers you're used to, like `mpicc`, `mpiicc`, `mpiifort`, etc.  

This guide will walk through how to utilize the Cray `PrgEnv-` environments with Cray MPICH, how to use "NLR-built" environments, and how to make sure your build is using the dependencies you expect.


### What is "PrgEnv-"?

Kestrel comes pre-packaged with several "programming environments." You can see which programming environments are available by typing `module load cpe-stack` and then `module avail PrgEnv`. For CPU codes, we focus on `PrgEnv-cray` and `PrgEnv-intel`. These environments provide compilers (accessible with the `cc`, `CC`, and `ftn` wrappers), Cray MPICH, and some other necessary lower-level libraries.

## The Tutorial

We're going to walk through building and running an MPI benchmarking code called IMB. This is a simple code that only requires a compiler and an MPI as dependencies (no scientific libraries, etc. are needed).

First, log onto Kestrel with
`ssh [your username]@kestrel.hpc.nlr.gov`

Let's grab an interactive node session:

`salloc -N 1 -n 104 --time=01:00:00 --account=<your allocation handle>`


### Environment 1: PrgEnv-cray 


Make a new directory
```
mkdir IMB-tutorial
cd IMB-tutorial
mkdir PrgEnv-cray
cd PrgEnv-cray
```

Then download the code:
```
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks
```

Now, we want to set up the environment. We can check our current environment with `module list`, which should output something like:

```
Currently Loaded Modules:
  1) Core/26.05   2) application/26.05   3) DefApps

```

In order to access the Cray programming environments, we first `module load cpe-stack`. Now, we can `module load PrgEnv-cray`, which will load the PrgEnv-cray environment and its associated modules, and `module load craype-x86-spr`, which is a module for targetting the kinds of CPU node processors that Kestrel has when building codes. If we run `module list` again we should see:

```
Currently Loaded Modules:
  1) Core/26.05          5) cce/19.0.0         9) craype-network-ofi   13) craype-x86-spr
  2) application/26.05   6) craype/2.7.34     10) cray-mpich/8.1.32
  3) DefApps             7) cray-dsmml/0.3.1  11) cray-libsci/25.03.0
  4) cpe-stack/25.03     8) libfabric/1.22.0  12) PrgEnv-cray/8.6.0
```


Now, we can build the code. From the `mpi-benchmarks` directory, run the command:

`CC=cc CXX=CC CXXFLAGS="-std=c++11" make IMB-MPI1`

What does this do?

`CC=cc` : set the c compiler to be `cc`. Recall that `cc` is the Cray wrapper around a c-compiler. Because we're in PrgEnv-cray, we expect the c compiler to be Cray's. We can test this by typing `cc --version`, which outputs:
```
[ohull@kl1 imb]$ cc --version
Cray clang version 19.0.0  (cc4d36e4ff3377d45f0e6e892b5dacd82009a0ca)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/cray/pe/cce/19.0.0/cce-clang/x86_64/bin
Build config: +assertions
```

As expected, we are using Cray's C compiler.

`CXX=CC`: This sets the C++ compiler to be `CC`, in the same way as `CC=cc` for the C compiler above.

`CXXFLAGS="-std=c++11"` tells the compiler to use the C++11 standard for compiling the C++ code, which is necessary because IMB has some code that is deprecated in C++17, which is the standard that Cray's C++ compiler defaults to.

Finally,

`make IMB-MPI1` builds IMB-MPI1, the IMB executable that we want.

Let's see what libraries we dynamically linked to in this build. Once the code is done building, type:
`ldd ./IMB-MPI1`

This will show all libraries required by the program (on the lefthand side) and the specific implementation of those libraries that the build is currently pointing to (on the righthand side).

Let's focus on MPI. Run:

`ldd ./IMB-MPI1 | grep mpi`

This should output something like:

```
[ohull@kl1 PrgEnv-cray]$ ldd IMB-MPI1 | grep mpi
	libmpi_cray.so.12 => /opt/cray/pe/lib64/libmpi_cray.so.12 (0x00007fddee9ea000)
```

So, the MPI library we're using is Cray's MPI (Cray MPICH)

Let's run the code:

`srun -N 1 -n 104 ./IMB-MPI1 AllReduce > out`

When it completes, take a look at the out file:

`cat out`

IMB swept from 1 MPI task to 104 MPI tasks, performing a number of MPI_ALLREDUCE calls between the MPI tasks (ranging from 0 bytes to 4194304 bytes)

**Note -- very important:** when you run IMB-MPI1, you *MUST* specify IMB-MPI1 as `./IMB-MPI1` or otherwise give a direct path to this specific version of `IMB-MPI1`. When we move to the NLR-built intel environment in this tutorial, we will have an `IMB-MPI1` already loaded into the path by default, and the command `srun IMB-MPI1` will execute the default `IMB-MPI1`, not the one you just built.


### Environment 2: PrgEnv-intel

We'll now repeat all the above steps, except now with PrgEnv-intel. Return to your `IMB-tutorial` directory, and `mkdir PrgEnv-intel`

Now, load the PrgEnv-intel environment by swapping it with the already-loaded PrgEnv-cray:

```
module swap PrgEnv-cray PrgEnv-intel
```

Again, we can test which C compiler we're using with:
`cc --version`
Now, this should output something like:
```
Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2 (2025.3.2.20260112)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /nopt/nlr/apps/kestrel-cpu/oneapi/2025.3.1/compiler/2025.3/bin/compiler
Configuration file: /nopt/nlr/apps/kestrel-cpu/oneapi/2025.3.1/compiler/2025.3/bin/compiler/../icx.cfg
```

Contrast this to when we ran `cc --version` in the PrgEnv-cray section. We're now using a different compiler (Intel oneAPI) under the hood.

We can now repeat the steps we took in the PrgEnv-cray section. Move up two directories and re-download the code:

```
cd ../../
mkdir PrgEnv-intel
cd PrgEnv-intel
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks
```

and build it:

`CC=cc CXX=CC CXXFLAGS="-std=c++11" make IMB-MPI1`

Note that we specify the same compiler wrapper, cc, to be the C compiler (the `CC=cc` part of the line above), as we did in the PrgEnv-cray section. But, `cc` now wraps around the intel-oneapi C compiler, instead of the Cray C compiler. So, we will be building with a different compiler, even though the build command is identical!

Again, we can run with:

`srun -N 1 -n 104 ./IMB-MPI1 AllReduce > out`

Or check which libraries are dynamically linked:

`ldd ./IMB-MPI1 `

Or, for MPI specifically:

```
[ohull@kl1 PrgEnv-intel]$ ldd ./IMB-MPI1 | grep mpi
	libmpi_intel.so.12 => /opt/cray/pe/lib64/libmpi_intel.so.12 (0x00007f13f8f8f000)
```

Note that this MPI library is indeed still Cray MPICH. The name is different than in the PrgEnv-cray section because it is specifically Cray MPICH built to be compatible with Intel compilers, not Cray compilers, as in the last example.

### Environment 3: Intel Compilers and Intel MPI

We've now seen two examples using Cray's environments, `PrgEnv-cray` and `PrgEnv-intel`. Let's build IMB using one of NLR's environments, which are separate from Cray's.

First, go back to your `IMB-tutorial` directory and re-clone the code:

```
cd ../../
mkdir intel-intelMPI
cd intel-intelMPI
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks 
```

Then, load the NLR environment. To do this, first run `module restore`.

Let's check out our options for Intel's oneapi toolchain:

`module avail oneapi`

This shows one `oneapi` module available. First, we load this module: `module load oneapi`. Now, we can re-run `module avail oneapi` and will see many more modules available, for example:

```
$ module avail oneapi

[ oneapi/2025.3.1 ]
   intel-oneapi-advisor/2025.4.1          intel-oneapi-ipp/2021.10.0  (D)
   intel-oneapi-dal/2023.2.0              intel-oneapi-ippcp/2021.8.0
   intel-oneapi-dal/2024.0.0       (D)    intel-oneapi-ippcp/2021.9.0 (D)
   intel-oneapi-dnn/2023.2.0              intel-oneapi-mkl/2023.2.0
   intel-oneapi-dnn/2024.0.0       (D)    intel-oneapi-mkl/2024.0.0
   intel-oneapi-dpl/2022.2.0              intel-oneapi-mkl/2025.3.1   (D)
   intel-oneapi-dpl/2022.3.0       (D)    intel-oneapi-mpi/2021.17.2
   intel-oneapi-inspector/2024.1.0        intel-oneapi-tbb/2021.10.0
   intel-oneapi-ipp/2021.9.0              intel-oneapi-vtune/2025.8.1
```

For this tutorial, we only need `intel-oneapi-mpi`. The above list shows many other toolchain modules associated with `oneapi`.

```
module load intel-oneapi-mpi
```

Now, we can build IMB with the intel compilers and Intel MPI:

`CC=mpiicx CXX=mpiicpx CXXFLAGS="-std=c++11" make IMB-MPI1`

Note that this command is different than the make commands we saw in the PrgEnv-cray and PrgEnv-intel sections.

Instead of `CC=cc` and `CXX=CC` we have `CC=mpiicx` and `CXX=mpiicpx`. `mpiicx`, is the intel MPI wrapper around the intel C compiler, and `mpiicpx` is the same but for C++.

Now, from the `your/path/to/IMB-tutorial/intel-intelMPI/mpi-benchmarks` directory, we can run the command:

`ldd ./IMB-MPI1 | grep mpi`

This outputs something like:

```
[ohull@kl1 intel-intelMPI]$ ldd ./IMB-MPI1 | grep mpi
	libmpicxx.so.12 => /nopt/nlr/apps/kestrel-cpu/spack/envs/sw26.05/opt/intel-oneapi-compilers-2025.3.2/intel-oneapi-mpi-2021.17.2-24hgtsxasvqwz7irtuvkg5e6m7rf7z2h/mpi/2021.17/lib/libmpicxx.so.12 (0x00007f0097b35000)
	libmpifort.so.12 => /nopt/nlr/apps/kestrel-cpu/spack/envs/sw26.05/opt/intel-oneapi-compilers-2025.3.2/intel-oneapi-mpi-2021.17.2-24hgtsxasvqwz7irtuvkg5e6m7rf7z2h/mpi/2021.17/lib/libmpifort.so.12 (0x00007f0097739000)
	libmpi.so.12 => /nopt/nlr/apps/kestrel-cpu/spack/envs/sw26.05/opt/intel-oneapi-compilers-2025.3.2/intel-oneapi-mpi-2021.17.2-24hgtsxasvqwz7irtuvkg5e6m7rf7z2h/mpi/2021.17/lib/libmpi.so.12 (0x00007f008d676000)
```

We see a few more libraries than we saw with the PrgEnvs. For example, we now have `libmpicxx`, `libmpifort`, and `libmpi`, instead of just `libmpi_intel` or `libmpi_cray`, as was the case with the two PrgEnvs. We can see that our three MPI library dependencies are pointing to the corresponding library's in the NLR-built environments.

## Final Words

Keeping track of the environments on Kestrel can be tricky at first. The key point to remember is that there are two separate "realms" of environments: first, the Cray environments, which are accessed by `module load cpe-stack`, and which typically use Cray MPICH. When using a `PrgEnv-*` toolchain, best practices dictate the use of the `cc`, `CC`, and `ftn` compiler wrappers for C, C++, and Fortran, respectively. The second "realm" of environments are the NLR-built environments that function similar to how the environments on Eagle functioned, and which use the more familiar compiler wrappers like `mpiicx` (for compiling C code with intel/intel MPI) or `mpicc` (for compiling C code with gcc/Open MPI.)
