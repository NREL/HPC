# Libraries How-To: Linking Scientific Libraries

This page is a tutorial explaining how to include scientific libraries when compiling software. 

There are a few common scientific libraries: LAPACK, BLAS, BLACS, scaLAPACK, FFTW, HDF5, and others. These libraries are generally highly optimized, and many scientific programs favor use of these libraries over in-house implementations of similar functionality. See our [libraries overview](index.md) page for more information.

Scientific libraries can be packaged together, like in the Intel Math Kernel Library (MKL), or Cray’s LibSci. They can also be built completely separately and act as standalone libraries. These libraries can be built with different MPI implementations and compiler choices. 

If you’re building a code that relies on one or more of these libraries, you can choose how to include these libraries. By the end of this tutorial, how to include these libraries should be clearer. If you need help building a particular package on an NLR machine, please contact [HPC help](mailto:hpc-help@nlr.gov). 

## Makefiles, autoconf, and cmake

Build tools like make, autoconf, and cmake are convenient ways to automate the compilation of a code. If you’re building a package, you may need to modify/customize how the code compiles, e.g., so it finds and includes the libraries you want. This may involve directly modifying the makefile, modifying the make.include (or make.inc, makefile.include, etc.) file, or using tools like autoconf or CMake to configure the makefile. 

Modifying a makefile (or make.include, etc.) so it compiles using the scientific libraries you want can be a daunting process. We’ll go through a prototypical example and show how different libraries can be included in the build of a program. To do this, we’ll use a makefile.include file for the electronic structure program VASP.

!!! note
	 We provide a walkthrough of linking scientific libraries using the VASP code as an example. This walkthrough tries to demonstrate key features of the general process of including scientific libraries in a build. We note that the exact build and modification process will vary between codes. Consulting the documentation of the code you’re trying to build is always the best place to start. 

## Walkthrough

### Overview

We’ll use the VASP makefile.include file as our walkthrough example. We can find a number of VASP makefile.include files [here](https://www.vasp.at/wiki/index.php/Makefile.include). We’ll be looking specifically at [this](https://www.vasp.at/wiki/index.php/Makefile.include.intel_omp) file.

We’ll take a look at building with Intel MKL and the HDF5 package. 

### Building with MKL and HDF5

We want to build with MKL and HDF5. If we look at the VASP documentation, we see that LAPACK, scaLAPACK, BLAS, and FFTW are required. MKL covers all of these needs. Thus, we need to tell the makefile where to look for MKL.

### Environment Preparation

We need our MKL to be built with the same compilers and MPI implementation as we’re building VASP with. Let’s see what sorts of MKL builds are available to us. Using the following command to show what versions of the oneapi toolchain are available:

` module avail oneapi` 

On Kestrel's Rhel9 CPU nodes, this yields the output `oneapi/2025.3.1` under the `Base Modules` category. We load this module with `module load oneapi/2025.3.1`. Now, we can run `module avail oneapi` again, and this time we see:

```
[ oneapi/2025.3.1 ]
   intel-oneapi-advisor/2025.4.1        intel-oneapi-dpl/2022.2.0              intel-oneapi-ippcp/2021.8.0        intel-oneapi-mpi/2021.17.2
   intel-oneapi-dal/2023.2.0            intel-oneapi-dpl/2022.3.0       (D)    intel-oneapi-ippcp/2021.9.0 (D)    intel-oneapi-tbb/2021.10.0
   intel-oneapi-dal/2024.0.0     (D)    intel-oneapi-inspector/2024.1.0        intel-oneapi-mkl/2023.2.0          intel-oneapi-vtune/2025.8.1
   intel-oneapi-dnn/2023.2.0            intel-oneapi-ipp/2021.9.0              intel-oneapi-mkl/2024.0.0
   intel-oneapi-dnn/2024.0.0     (D)    intel-oneapi-ipp/2021.10.0      (D)    intel-oneapi-mkl/2025.3.1   (D)

[ Base Modules ]
   oneapi/2025.3.1 (L)
```

In general, we will want to use the oneapi/2025.3.1 toolchain as much as possible. To “use the Intel oneapi toolchain” means to use Intel compilers and Intel’s implementation of MPI to compile VASP. We’re doing this because mkl was built with this toolchain, and we want our toolchains to match as best as possible to minimize build errors and bugs. 

In the above `module avail` output, we can locate the oneapi modules we want, and load each of them: 

``` 
module load intel-oneapi-mkl/2025.3.1
module load intel-oneapi-mpi/2021.17.2
``` 

How do we know these are the ones we want? The first line loads the mkl module, which will make the desired mkl libraries available to us. The second line gives us mpiifort, the Intel MPI fortran compiler. (You can test with `which mpiifort`, which should produce a path that includes `intel-oneapi-mpi`)

In general, if you want to build your own version of an application but aren't sure which toolchain modules to use, you can check how existing applications were setup using `module show`. For example with VASP, `module show vasp` on Rhel9 produces the following towards the bottom of the output:

```
load("cpe-stack/25.03")
load("craype-x86-spr")
load("oneapi")
load("intel-oneapi-mpi")
load("intel-oneapi-mkl")
load("intel-oneapi-tbb")
load("hdf5")
setenv("I_MPI_FABRICS","shm")
prepend_path("PATH","/nopt/nlr/apps/kestrel-cpu/software/vasp/binaries/vasp.6.5.1_openMP+hdf5+XClibs")
```

You can follow these load commands by replacing with their equivalent command-line commands. E.g., from a clean environment (for example, when you first log in or after using `module restore`):

```
module load cpe-stack/25.03
module load craype-x86-sor
module load oneapi
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load intel-oneapi-tbb
module load hdf5
export I_MPI_FABRICS=shm
PATH=/nopt/nlr/apps/kestrel-cpu/software/vasp/binaries/vasp.6.5.1_openMP+hdf5+XClibs:$PATH
```

Would reproduce the environment set up that occurs when you simply `module load vasp`. This gives a good starting point for setting up an environment that avoids version conflicts and other problems that arise from trying to find sets of modules that work together without error.

### Modifying the Makefile for MKL

Now that we have the toolchain loaded into our environment, let’s take a look at the actual makefile.include file (link to file [here](https://www.vasp.at/wiki/index.php/Makefile.include.intel_omp)). There are two important sections for the purpose of getting the code to build. The first: 

```
CPP         = fpp -f_com=no -free -w0  $*$(FUFFIX) $*$(SUFFIX) $(CPP_OPTIONS) 
FC          = mpiifort -qopenmp 
FCL         = mpiifort 
```

The first line says that the compiler pre-processor will be fpp (try `which fpp` and you should get an output that shows `fpp` is in your path and coming from `oneapi`).

The second and third lines say that we’ll be using Intel’s MPI (Try `which mpiifort` to confirm that it is in your path). FC is the “Fortran Compiler” and FCL is the corresponding linker. The `-qopenmp` flag additionally says we’ll be compiling with openmp. Different compilers have different executable names (e.g. mpiifort for Intel MPI fortran compiler, mpifort for GNU). See the [Fortran documentation page](../../Languages/Fortran/) for a complete list. 

The next important section is given below: 

```  
# Intel MKL (FFTW, BLAS, LAPACK, and scaLAPACK) 
# (Note: for Intel Parallel Studio's MKL use -mkl instead of -qmkl) 
FCL        += -qmkl 
MKLROOT    ?= /path/to/your/mkl/installation 
LLIBS      += -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
INCS        =-I$(MKLROOT)/include/fftw 
```

This makefile.include file has been provided to us by VASP. Our job here is two-fold:

1. To ensure that we tell `make` (via the makefile.include file) the correct place to find MKL, I.e., to ensure that `MKLROOT` in the makefile.include file is set correctly.
2. To ensure that we tell `make` the correct libraries to reference within `MKLROOT`.

To do step 1, first type:

`module list` 

To see the modules you’ve loaded into your environment. You should have a version of `intel-oneapi-mkl` in the list.  If not, review the [environment preparation](#environment-preparation) section. Now, we use the `module show` command to find the root directory of mkl: 

`module show intel-oneapi-mkl` 

We see in the output of this command the following line: 

`setenv("MKLROOT", [path name])` 

where `[path name]` corresponds to the directory location of `MKLROOT`. 

If we type `echo $MKLROOT`, we can confirm that this environment variable is properly set from when we ran the command `module load intel-oneapi-mkl`. In the VASP makefile, we have `MKLROOT    ?= /path/to/your/mkl/installation`. The ?= means that this variable will not be set if `MKLROOT` has already been set. So, we can ignore this line if we’d like. However, to be safe, we can simply copy the path of the MKL root directory to this line in makefile.include, so that this line now reads: 

`MKLROOT    ?= [actual path for MKLROOT]` 

where `[actual path for MKLROOT]` is an exact copy/paste of the output of `echo $MKLROOT`.

!!! tip  
	The name of the environment variable for mkl’s root directory set by its module (`MKLROOT`, set when we `module load intel-oneapi-mkl`) is not necessarily going to match the corresponding root directory variable in a given makefile. It did in this instance, but that’s not guaranteed. The VASP makefile.include could have just as easily used `MKL_ROOT`, instead of `MKLROOT`. This is one reason why it’s safer to use `module show` to find the path of the root directory, then copy this path into the makefile, rather than rely on environment variables. Copying directly also encodes information about which exact version you used to build the code.

To do step 2, we should first look at the contents of `$MKLROOT`. To show the contents of the MKL directory, type

`ls $MKLROOT`

We should obtain an output that includes the folllowing directories (others may also be present):

`bin  env  etc  include  lib  share`

If we look closely at the makefile, we see beneath the `MKLROOT` line the following:

```
LLIBS      += -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
```

the `LLIBS` line is telling `make` which libraries in particular to pick out. 

So, we want to go into the lib directory, and then the intel64 directory (since LLIBS is pointing to `$MKLROOT/lib/intel64`). Let's see what's inside with the `ls` command:

`ls  $MKLROOT/lib/intel64`

There's a lot of stuff in this directory! VASP helps us by telling us we need the `mkl_scalapack_lp64` and `mkl_blacs_openmpi_lp64` libraries specifically. You won't always be told exactly which libraries, and figuring this out, if the information is not provided to you in the package documentation, can require some tinkering.

In general, the `.a` extension is for static linking, and the `.so` extension is for dynamic linking. For MKL in particular, the part `ilp64` vs `lp64` refer to two different interfaces to the MKL library. 

!!! tip
	Notice that, inside `$MKLROOT/lib/intel64`, the  filenames all start with `libmkl`, but in our makefile, we reference `lmkl_scalapack_lp64`. That's not a file in `$MKLROOT/lib/intel64`, but `libmkl_scalapack_lp64.so` is. The notation is that "big L" references the directories that the libraries are in, and the "little l" references the particular libraries. For example:
	<pre> LLIBS += <b>-L</b>$(MKLROOT)/lib/intel64 </pre>
	<pre> <b>-l</b>mkl_scalapack_lp64</pre> This is just a convention, but is important to get right because your compile will fail otherwise.

Now that we have verified we have the correct `MKLROOT` set in the makefile.include, and we have an idea about how it's referencing the libraries within, we can move on to linking the HDF5 library.

### Modifying the Makefile for HDF5

Because HDF5 is an optional library, we could compile the code now if we wanted to. However, for the sake of practice, let’s uncomment the block in the makefile.include file related to HDF5 and repeat the exercise of linking a library: 

``` 
# HDF5-support (optional but strongly recommended) 
CPP_OPTIONS+= -DVASP_HDF5 
HDF5_ROOT  ?= /path/to/your/hdf5/installation 
LLIBS      += -L$(HDF5_ROOT)/lib -lhdf5_fortran 
INCS       += -I$(HDF5_ROOT)/include 
``` 

Our job, again, is to give the makefile the correct directions to our library. In this case, it’s HDF5. We already loaded hdf5 in our environment preparation step, so now we can inspect it with `module show hdf5`. Again, we must locate its root directory. In the `module show hdf5` output, we see the line for setting the HDF5 root directory environment variable: 

`setenv("NREL_HDF5_ROOT", [path to hdf5]` 

Like before, we copy this path into our makefile.include, copy/pasting the output of `echo $NREL_HDF5_ROOT`

```
# HDF5-support (optional but strongly recommended) 
CPP_OPTIONS+= -DVASP_HDF5 
HDF5_ROOT  ?= [actual path for NREL_HDF5_ROOT]
LLIBS      += -L$(HDF5_ROOT)/lib -lhdf5_fortran 
INCS       += -I$(HDF5_ROOT)/include 
```

We can confirm that the library VASP is looking for, `libhdf5_fortran.so` (since the LLIBS line has `-lhdf5_fortran`), is present by running `ls $NREL_HDF5_ROOT/lib`

We’re ready to compile! In the case of VASP, the compile command is `make DEPS=1 std` but in general, the command may be `make all` or similar (consult the documentation of the code you’re trying to build). 

If you’re working with a code that has a testsuite, now is a good time to run the testsuite to make sure that your compile was successful. 

## Summary of Steps

1. Download the source code of the package you’re trying to build. This will generally be found on the website of the package. 
2. Consult the documentation of the package to find out what scientific libraries are needed, and if the package developers provide guidance on what toolchains/libraries are best 
3. Determine the availability of the needed scientific libraries.  
    1. Can a “library-of-libraries” like MKL or LibSci be used? 
    2. Does NLR support the library as a module?  
        1. If so, determine the toolchain it was built with (usually given in the name of the module). If the toolchain is not clear from the name of the module, try the `ldd` command (e.g., `ldd path/to/executable/executable`), which will show you the dynamically linked libraries of the executable.
4. Prepare your environment 
    1. `module load` the necessary modules to prepare your environment. (See  [environment preparation](#environment-preparation) step of VASP example) 
5. Prepare your makefile 
    1. Make sure that the compilers and (optional) MPI used in the makefile match what is used to build your scientific libraries as best as possible 
    2. Make sure that the paths to the scientific libraries in the makefile match the path given by the `module show` command 
    3. Make sure the proper “little L” libraries are referenced in the makefile 
6. Compile!


## Questions?

If you’re still stuck and unable to successfully link the scientific libraries you need, get in contact with [HPC help](mailto:hpc-help@nlr.gov).
