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

None

### Terms
* `JULIA_HOME` is the base directory of julia source code (initially called `julia` after `git clone`)

### Instructions

1. Load the following modules:
    * gcc -- tested with gcc/8.2.0
    * mkl (optional) -- tested with mkl/2018.3.222 and mkl/2019.5.281
    * conda -- tested with conda/5.3
2. Get the Julia source code 
`git clone https://github.com/JuliaLang/julia.git`
4. Change to the version of Julia you want to build `git checkout <julia_version>`
5. In `Make.user` (you will need to create the file if it doesn't exist) in `JULIA_HOME` put the following:
    * *If you want to compile Julia **with** MKL*
        * `USE_INTEL_MKL=1` -- Use Intel versions of BLAS and LAPACK (this is why we loaded mkl module)
        * `USE_BLAS64=0` -- Use the 64-bit library with the 32-bit integer interface (this will necessitate changes in `Make.inc`)
        * `USE_BINARYBUILDER=0` -- Build required dependencies from source (gets around known bug with linking and patching that is a problem with ld version 2.25)
    * *If you want to compile Julia **without** MKL* (OpenBLAS will be used for BLAS and LAPACK libraries)
        * `USE_INTEL_MKL=0`  -- Download, build and use openblas for BLAS/LAPACK libraries
        * `USE_BINARYBUILDER=0` -- Build required dependencies from source—gets around known bug with linking and patching that is a problem with ld version 2.25
    * **NOTE**: I found it useful to create the file `Make.user` in another location (e.g. home directory) and drop a link into the Julia build directory as I used `git clean -x -f -d` to make sure everything is completely clean
    * **NOTE**: If you are going to be doing significant matrix-vector operations directly in Julia, then you will want to compile it with MKL. If most of the matrix-vector operations are being done in a subprogram or library (e.g. Ipopt) then it will make no difference what you compile Julia with.  In this latter case, it is recommended that you compile with OPENBLAS since that is easier.
    * **NOTE**: When compiling Julia with MKL, Julia uses the `single dynamic library` option for linking.  Any dynamic libraries (e.g. ipopt or coinhsl) loaded by Julia also need to be linked to MKL with this approach.  Failing to do so will result in unusual behavior (like getting garbage values passed to the MKL function calls).
6. (If compiling Julia **with** MKL otherwise skip to step 6) Open `Make.inc` in your favorite editor and make the following change
    * find where `MKLLIB` is set (there will be an if-else statement depending on the value of `USE_BLAS64`)
    * change the else clause to read `MKLLIB := $(MKLROOT)/lib/intel64`
7. `cd JULIA_HOME/deps`
8. `make extract-suitesparse`
    * **NOTE**: if this produces the message 'nothing to be done for target `make extract-suitesparse`' (or something like it), this means you didn’t put `Make.user` in `JULIA_HOME` or `USE_BINARAYBUILD` is set incorrectly in `Make.user`
9. `cd scratch/SuiteSparse-5.4.0/UMFPACK/Lib/`
10. open `Makefile` in your favorite editor and make the following change:
    * do a global replace on `USER` in the `Makefile`--i.e. change all occurrences of the variable `USER` to something else like `MUSER`
    * this `Makefile` defines a `USER` variable that leads to problems with `xalt/ld` (a script that invokes ld)
11. `cd JULIA_HOME`
12. `make -j 4` -- `-j 4` allows `make` to use 4 processes to build and can speed up compilation (additional speed ups may be possible by increasing the number of processes)
