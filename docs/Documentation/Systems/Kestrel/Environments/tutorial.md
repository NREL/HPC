# Environments tutorial

In this tutorial, we will walk through how to build and run a basic MPI code using the four principal toolchains/software stacks on Kestrel. We will discuss common pitfalls in building and running within each of these toolchains, too.

We summarize these toolchains in the below table:

| PrgEnv-* | Compiler | MPI |
|----------|----------|--------------------|
|cray| cray cce| Cray MPICH |
|intel| intel | Cray MPICH | 
|n/a| intel | Intel MPI | 
|n/a| gcc | Open MPI | 

**Note**: There is an option to compile with MPICH-based MPI (e.g., Intel MPI but not Open MPI) and then use the module `cray-mpich-abi` at run-time, which causes the code to use Cray MPICH instead of the MPI it was built with. More information on how to use this feature will be added soon.

## Introduction

Kestrel is a Cray machine whose nodes are connected by "Cray Slingshot" (contrast this to Eagle, which uses infiniband). We've found that packages that make use of Cray tools like Cray MPICH perform faster than when the same package is built and run without Cray tools (e.g. compiling and running with intel MPI), in part because these Cray tools are optimized to work well with Cray Slingshot.

Most of us coming from Eagle are probably used to running our codes with Intel MPI or Open MPI, but not Cray MPICH.

Using the cray-designed programming environments ("PrgEnvs") requires using special Cray compiler wrappers `cc` and `ftn`. These wrappers replace the MPI compiler wrappers you're used to, like `mpicc`, `mpiicc`, `mpiifort`, etc.  

This guide will walk through how to utilize the Cray `PrgEnv-` environments with Cray MPICH, how to use "NREL-built" environments, and how to make sure your build is using the dependencies you expect.


### What is "PrgEnv-"?

Kestrel comes pre-packaged with several "programming environments." You can see which programming environments are available by typing `module avail PrgEnv`. For CPU codes, we focus on `PrgEnv-cray` and `PrgEnv-intel`. These environments provide compilers (accessible with the `cc`, `CC`, and `ftn` wrappers), Cray MPICH, and some other necessary lower-level libraries.

## The Tutorial

We're going to walk through building and running an MPI benchmarking code called IMB. This is a simple code that only requires a compiler and an MPI as dependencies (no scientific libraries, etc. are needed).

First, log onto Kestrel with
`ssh [your username]@kestrel.hpc.nrel.gov`

Let's grab an interactive node session:

`salloc -N 1 -n 104 --time=01:00:00 --account=[your account name]`


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

PrgEnv-cray is the default environment on Kestrel, so it should already be loaded upon login to Kestrel. To check, type `module list` and make sure you see `PrgEnv-cray` somewhere in the module list. If you don't, you can restore the default environment (PrgEnv-cray) by simply running `module restore`.

Now, we can build the code. Run the command:

`CC=cc CXX=CC CXXFLAGS="-std=c++11" make IMB-MPI1`

What does this do?

`CC=cc` : set the c compiler to be `cc`. Recall that `cc` is the Cray wrapper around a c-compiler. Because we're in PrgEnv-cray, we expect the c compiler to be Cray's. We can test this by typing `cc --version`, which outputs:
```
[ohull@kl1 imb]$ cc --version
No supported cpu target is set, CRAY_CPU_TARGET=x86-64 will be used.
Load a valid targeting module or set CRAY_CPU_TARGET
Cray clang version 14.0.4  (3d8a48c51d4c92570b90f8f94df80601b08918b8)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/cray/pe/cce/14.0.4/cce-clang/x86_64/share/../bin
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

**Note -- very important:** when you run IMB-MPI1, you *MUST* specify IMB-MPI1 as `./IMB-MPI1` or otherwise give a direct path to this specific version of `IMB-MPI1`. When we move to the NREL-built intel environment in this tutorial, we will have an `IMB-MPI1` already loaded into the path by default, and the command `srun IMB-MPI1` will execute the default `IMB-MPI1`, not the one you just built.

If you'd like, you can also submit this as a slurm job. Make a file `submit-IMB.in`, and paste the following contents:

```
#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=104

#!/bin/bash

srun -N 1 --tasks-per-node=104 --mpi=pmi2 your/path/to/IMB-tutorial/PrgEnv-cray/mpi-benchmarks/IMB-MPI1 Allreduce > out
```

Don't forget to update `your/path/to/IMB-tutorial/PrgEnv-cray/mpi-benchmarks/IMB-MPI1` to the actual path to your IMB-MPI1 executable.  

Then, `sbatch submit-IMB.in`

### Environment 2: PrgEnv-intel

We'll now repeat all the above steps, except now with PrgEnv-intel. Return to your `IMB-tutorial` directory, and `mkdir PrgEnv-intel`

Now, load the PrgEnv-intel environment:

```
module restore
module swap PrgEnv-cray PrgEnv-intel
module unload cray-libsci
```

Note that where possible, we want to avoid using `module purge` because it can unset some environment variables that we generally want to keep. So, instead we run `module restore` to restore the default environment (PrgEnv-cray) and then swap from PrgEnv-cray to PrgEnv-intel with `module swap PrgEnv-cray PrgEnv-intel`. Finally, we unload the `cray-libsci` package for the sake of simplicity (as of 4/23/24, we are working through resolving a default versioning conflict between cray-libsci and PrgEnv-intel. If you need to use cray-libsci within PrgEnv-intel, please reach out to hpc-help@nrel.gov)

Again, we can test which C compiler we're using with:
`cc --version`
Now, this should output something like:
```
[ohull@x1000c0s0b0n0 mpi-benchmarks]$ cc --version
Intel(R) oneAPI DPC++/C++ Compiler 2023.2.0 (2023.2.0.20230622)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /nopt/nrel/apps/cpu_stack/compilers/02-24/spack/opt/spack/linux-rhel8-sapphirerapids/gcc-12.2.1/intel-oneapi-compilers-2023.2.0-hwdq5hei2obxznfjhtlav4mi5h5jd4zw/compiler/2023.2.0/linux/bin-llvm
Configuration file: /nopt/nrel/apps/cpu_stack/compilers/02-24/spack/opt/spack/linux-rhel8-sapphirerapids/gcc-12.2.1/intel-oneapi-compilers-2023.2.0-hwdq5hei2obxznfjhtlav4mi5h5jd4zw/compiler/2023.2.0/linux/bin-llvm/../bin/icx.cfg
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

`srun -N 1 -n 104 --mpi=pmi2 ./IMB-MPI1 AllReduce > out`

Or check which libraries are dynamically linked:

`ldd ./IMB-MPI1 `

Or, for MPI specifically:

```
[ohull@kl1 PrgEnv-intel]$ ldd ./IMB-MPI1 | grep mpi
	libmpi_intel.so.12 => /opt/cray/pe/lib64/libmpi_intel.so.12 (0x00007f13f8f8f000)
```

Note that this MPI library is indeed still Cray MPICH, the name is different than in the PrgEnv-cray section because iti is specifically Cray MPICH built to be compatible with intel compilers, not cray compilers, as in the last example.

You can also submit this inside a Slurm submit script:

```
#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=104
#SBATCH --account=<your allocation handle>

#!/bin/bash

module restore
module swap PrgEnv-cray PrgEnv-intel
module unload cray-libsci

srun -N 1 --tasks-per-node=104 --mpi=pmi2 your/path/to/IMB-tutorial/PrgEnv-intel/mpi-benchmarks/IMB-MPI1 Allreduce > out
```

Note that the only difference between this submit script and the one for Environment 1 is that we exchange `PrgEnv-cray` for `PrgEnv-intel`.

### Environment 3: Intel Compilers and Intel MPI

We've now seen two examples using Cray's environments, `PrgEnv-cray` and `PrgEnv-intel`. Let's build IMB using one of NREL's environments, which are separate from Cray's.

First, go back to your `IMB-tutorial` directory and re-clone the code:

```
cd ../../
mkdir intel-intelMPI
cd intel-intelMPI
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks 
```

Then, load the NREL environment. To do this, first run:
```
module restore
module unload PrgEnv-cray
```

Again, we want to avoid `module purge` where possible, so we restore the environment to default (PrgEnv-cray) and then unload the default environment, in order to retain underlying environment variables.


Let's check out our options for Intel compilers now:

`module avail intel`

We should see a number of modules. Some correspond to applications built with an intel toolchain (e.g. `amr-wind/main-intel-oneapi-mpi-intel`, whose name implies that amr-wind was built with the intel oneapi MPI and intel compilers). Others correspond to the MPI (e.g. `intel-oneapi-mpi/2021.8.0-intel`) or the compilers itself (e.g. `intel-oneapi-compilers/2022.1.0`)

Let's load Intel MPI and Intel compilers:

```
module load intel-oneapi
module load intel-oneapi-compilers
module load intel-oneapi-mpi
```

Note that if we look back at `module avail intel` and look at the header above, e.g., `intel-oneapi`, we can see that these intel modules live in `/nopt/nrel/apps/cpu_stack/modules/default/compilers_mpi` -- this is different than the PrgEnvs, who can be found in `/opt/cray/pe/lmod/modulefiles/core`. This is one way to tell that you are using NREL's set of modules and not Cray's set of modules.

 Now, we can build IMB with the intel compilers and Intel MPI:

`CC=mpiicc CXX=mpiicpc CXXFLAGS="-std=c++11" make IMB-MPI1`

Note that this command is slightly different than the make commands we saw in the PrgEnv-cray and PrgEnv-intel sections.

Instead of `CC=cc` and `CXX=CC` we have `CC=mpiicc` and `CXX=mpiicpc`. `mpiicc`, is the intel MPI wrapper around the intel C compiler, and `mpiicpc` is the same but for C++.

Remember that warning about `IMB-MPI1` being in the default path? This is now true, so be careful that when you run the package, you're running the version you just built, NOT the default path version.

If you're still inside `your/path/to/IMB-tutorial/intel-intelMPI/mpi-benchmarks` then we can run the command:

`ldd ./IMB-MPI1 | grep mpi`

This outputs something like:

```
[ohull@kl1 intel-intelMPI]$ ldd ./IMB-MPI1 | grep mpi
	libmpicxx.so.12 => /nopt/nrel/apps/mpi/07-23/spack/opt/spack/linux-rhel8-icelake/intel-2021.6.0/intel-oneapi-mpi-2021.8.0-6pnag4mmmx6lvoczign5a4fslwvbgebb/mpi/2021.8.0/lib/libmpicxx.so.12 (0x00007f94e5e09000)
	libmpifort.so.12 => /nopt/nrel/apps/mpi/07-23/spack/opt/spack/linux-rhel8-icelake/intel-2021.6.0/intel-oneapi-mpi-2021.8.0-6pnag4mmmx6lvoczign5a4fslwvbgebb/mpi/2021.8.0/lib/libmpifort.so.12 (0x00007f94e5a55000)
	libmpi.so.12 => /nopt/nrel/apps/mpi/07-23/spack/opt/spack/linux-rhel8-icelake/intel-2021.6.0/intel-oneapi-mpi-2021.8.0-6pnag4mmmx6lvoczign5a4fslwvbgebb/mpi/2021.8.0/lib/release/libmpi.so.12 (0x00007f94e4138000)
```

We see a few more libraries than we saw with the PrgEnvs. For example, we now have `libmpicxx`, `libmpifort`, and `libmpi`, instead of just `libmpi_intel` or `libmpi_cray`, as was the case with the two PrgEnvs. We can see that our three MPI library dependencies are pointing to the corresponding library's in the NREL-built environments.

We can submit an IMB job with the following slurm script:

```
#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=104

module restore
module unload PrgEnv-cray

module load intel-oneapi
module load intel-oneapi-compilers
module load intel-oneapi-mpi

srun -N 1 --tasks-per-node=104  /your/path/to/IMB-tutorial/intel-intelMPI/mpi-benchmarks/IMB-MPI1 Allreduce > out
```

don't forget to replace `/your/path/to/IMB-tutorial/intel-intelMPI/mpi-benchmarks/IMB-MPI1` with your actual path.

### Environment 4: GCC and OpenMPI

Environment 4 works similarly to Environment 3, except instead of using the NREL-built intel modules, we'll use GCC and OpenMPI instead. Note that OpenMPI is not ever recommended to use multi-node, because it is unstable on cray slingshot networks. You should only use OpenMPI for single-node jobs.


Return to your `IMB-tutorial` directory and set up for gcc-openMPI:

```
cd ../../
mkdir gcc-openMPI
cd gcc-openMPI
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks 
```

Run:

```
module restore
module unload PrgEnv-cray
module unload cce
```

Note that unlike the NREL-intel case, loading `gcc` doesn't automatically unload `cce` ("cray compiler environment") so we do it manually here with `module unload cce`

Now, we can `module avail openmpi` to find openmpi-related modules. Then, load the version of openmpi that was built with gcc:

`module load openmpi/4.1.5-gcc`

And finally, load gcc. To see which versions of gcc are available, type `module avail gcc`. We'll use GCC 10: `module load gcc/10.1.0`

Now, we can build the code. Run the command:

`CC=mpicc CXX=mpic++ CXXFLAGS="-std=c++11" make IMB-MPI1`

Similar to using mpiicc and mpiicpc in the Environment 3 section, now we use mpicc and mpic++, because these are the Open MPI wrappers around the GCC C and C++ compilers (respectively). We are not using the `cc` and `CC` wrappers now because we are not using a `PrgEnv`. 

Once the executable is built, check the mpi library it's using with ldd:

`ldd ./IMB-MPI1 | grep libmpi`

This command should return something like:

```
[ohull@x1007c7s7b0n0 mpi-benchmarks]$ ldd ./IMB-MPI1 | grep libmpi
	libmpi.so.40 => /nopt/nrel/apps/mpi/07-23/spack/opt/spack/linux-rhel8-icelake/gcc-10.1.0/openmpi-4.1.5-s5tpzjd3y4scuw76cngwz44nuup6knjt/lib/libmpi.so.40 (0x00007f5e0c823000)
```

We see that libmpi is indeed pointing where we want it to: to the openmpi version of libmpi built with gcc-10.1.0.

Finally, we can submit an IMB job with the following slurm script:

```
#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=104

module restore
module unload PrgEnv-cray
module unload cce

module load openmpi/4.1.5-gcc
module load gcc/10.1.0

srun -N 1 --tasks-per-node=104 /your/path/to/IMB-tutorial/gcc-openMPI/mpi-benchmarks/IMB-MPI1 Allreduce > out
```

don't forget to replace `/your/path/to/IMB-tutorial/gcc-openMPI/mpi-benchmarks/IMB-MPI1` with your actual path.

## Final Words

With all four environments built, you could now run a few benchmarks comparing how MPI performs between them. Try this using 1 node and using 2 nodes, and compare the results for each environment. You should see that performance between all four environments is competitive on 1 node, but the two `PrgEnv` builds run a bit faster for large message sizes on 2 nodes, and the gcc/openmpi build is liable to randomly fail in the 2 node case.

Keeping track of the environments on Kestrel can be tricky at first. The key point to remember is that there are two separate "realms" of environments: the Cray `PrgEnv`s, which use Cray MPICH and best practices dictate the use of the `cc`, `CC`, and `ftn` compiler wrappers for C, C++, and Fortran, respectively, and the NREL-built environments that function similar to how the environments on Eagle function, and which use the more familiar compiler wrappers like `mpiicc` (for compiling C code with intel/intel MPI) or `mpicc` (for compiling C code with gcc/Open MPI.)

Earlier in the article, we mentioned the existence of the `cray-mpich-abi`, which allows you to compile your code with a non-Cray MPICH-based MPI, like Intel MPI, and then run the code with Cray MPICH via use of the `cray-mpich-abi` module. We will include instructions for how to use this in an updated version of the tutorial.
