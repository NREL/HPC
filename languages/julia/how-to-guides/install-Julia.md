# Installing Julia on Eagle

## Anaconda
Older versions of Julia are available from conda-forge channel:
```
conda create -n julia-env
source activate julia-env
conda install -c conda-forge julia
```

## Spack Build

### Prerequisites

A working version of Spack.  For detailed instructions on getting spack setup see the [github repository](https://github.com/spack/spack).  Briefly, this can be done with the following:
```
git clone https://github.com/spack/spack.git
cd spack
git checkout releases/v0.15 # Change to desired release
. share/spack/setup-env.sh # Activate spack shell support
```

### Instructions

**NOTE:** Steps 1 and 2 may be skipped when using the develop branch or any release branch after v0.15.

1. In the spack repository, open the file `var/spack/repos/builtin/packages/julia/package.py` in your favorite editor.
2. There is an if-else statement under the if statement 
```
if spec.target.family == 'x86_64'  or spec.target.family == 'x86':
```
Change the else clause to read
```
else:
    target_str = str(spec.target).replace('_','-')
    options += [
        'MARCH={0}'.format(target_str),
        'JULIA_CPU_TARGET={0}'.format(target_str)
    ]
```
3. Now install julia with spack
 ```
 spack install julia
 ```

## Do It Yourself Build (v 1.2 or later)

### Prerequisites

All the [required build tools and libraries](https://github.com/JuliaLang/julia/blob/master/doc/build/build.md#required-build-tools-and-external-libraries) are available on Eagle either by default or through modules.  The needed modules are covered in the instructions.

### Terms
* `JULIA_HOME` is the base directory of julia source code (initially called `julia` after `git clone`)

### Instructions
When compiling Julia you can choose to compile against Intel's MKL libraries or OpenBLAS for the Julia linear algebra operations. If you are going to be doing significant matrix-vector operations directly in Julia, then you will want to compile it with MKL. If most of the matrix-vector operations are being done in a subprogram or library (e.g. Ipopt) then it will make no difference what you compile Julia with.  In this latter case, it is recommended that you compile with OpenBLAS since that is significantly easier. Instructions for both choices are given below.

**NOTE**: When compiling Julia **with** MKL, Julia uses the `single dynamic library` option for linking.  Any dynamic libraries (e.g. ipopt or coinhsl) loaded by Julia also need to be linked to MKL with this approach.  Failing to do so will result in unusual behavior (like getting garbage values passed to the MKL function calls).

1. Load the following modules:
    * gcc (>= 5.1)
    * cmake (>= 3.4.3)
    * mkl (any version -- optional)
2. Get the Julia source code 
`git clone https://github.com/JuliaLang/julia.git`
3. `cd julia`
4. Change to the version of Julia you want to build `git checkout <julia_version>`
5. In `Make.user` (you will need to create the file if it doesn't exist) in `JULIA_HOME` put the following:
	* `MARCH=skylake-avx512` -- tell the compiler to [optimize floating point instructions for Eagle's Skylake processors](https://www.nrel.gov/hpc/eagle-software-libraries-mkl.html)
    * *If you want to compile Julia **with** MKL also add the following*
        * `USE_INTEL_MKL=1` -- Use Intel versions of BLAS and LAPACK (this is why we loaded mkl module)
        * `USE_BLAS64=0` -- Use the 64-bit library with the 32-bit integer interface. This will necessitate changes in `Make.inc`. The reasons for this are discussed in step 7.
    * **NOTE**: I found it useful to create the file `Make.user` in another location (e.g. home directory) and drop a link into the Julia build directory as I used `git clean -x -f -d` to make sure everything is completely clean
6. (Skip to step 8 if compiling Julia **without** MKL.) There are a couple of problems to overcome when compiling Julia with MKL.  The first is that a makefile in the SuiteSparse library package defines a `USER` variable that leads to problems with `xalt/ld` (a script that invokes ld).  To fix this do the following:
    * In JULIA_HOME fetch and unpack the SuiteSparse libraries
`make -C deps/ extract-suitesparse`
    * With your favorite editor, open the file
`JULIA_HOME/deps/scratch/SuiteSparse-5.4.0/UMFPACK/Lib/Makefile`
    * In the `Makefile`, do a global replace on `USER` --i.e. change all occurrences of the variable  `USER`  to something else like  `MUSER`
7. The second problem is that when compiling against MKL, Julia either uses the 32-bit MKL libraries or the 64-bit MKL libraries with *64-bit interface*.  It is common for other libraries (e.g. Ipopt or HSL) to compile against the 64-bit MKL libraries with *32-bit interface*.  This causes unusual behavior.  To make Julia compile against the 64-bit MKL libraries with 32-bit interface, do the following:
    * Open `Make.inc` in your favorite editor and make the following change
        * find where `MKLLIB` is set (there will be an if-else statement depending on the value of `USE_BLAS64`)
        * change the else clause to read `MKLLIB := $(MKLROOT)/lib/intel64`
8. `make -j4` -- `-j4` allows `make` to use 4 processes to build and can speed up compilation (additional speed ups may be possible by increasing the number of processes)
