
# Obtaining HSL Source Code

Go to the [HSL Ipopt site](http://www.hsl.rl.ac.uk/ipopt/) and click on one of the links on the right below either "Coin-HSL Full (RC)" or "Coin-HSL Full (Stable)" that says "source".  After filling out the form for a personal license, HSL will email you a link to download the source code.  This normally takes about a day.

Notes about the available solvers are also given at the above link.  It can be easily found again by googling "HSL for Ipopt".

# Eagle

The default version of Ipopt distributed with Ipopt.jl on linux links to the openblas library.  This causes issues when linking the HSL library to the MKL libraries.  For this reason, to use HSL linear solvers with Ipopt on Eagle, either we must compile Ipopt from scratch or compile HSL with openblas instead of MKL.  For performance reasons, we have elected to compile Ipopt from scratch so that we can use the MKL libraries.

The following provides detailed instructions for compiling Ipopt with HSL and Mumps on Eagle.  

### Environment

We will make use of the following environment variables.

```
# Location of metis.h
export METIS_HEADER=${HOME}/.conda-envs/<conda_environment>/include
# Location of metis library
export METIS_LIBRARY=${HOME}/.conda-envs/<conda_environment>/lib

# Directory for keeping source code and build products
export MYAPPS=${HOME}/apps
# Location of static and dynamic libraries
export MYLIB=${MYAPPS}/lib
```

These can be added to the .bash_profile file (or equivalent for other shells).  Remember after adding these to source .bash_profile (or equivalent) or to open a new terminal and do all building there.

### Prerequisites

As noted in the [Ipopt install instructions](https://coin-or.github.io/Ipopt/INSTALL.html), we will need `pkg-config` as well as a C and Fortran compiler.  `pkg-config` is available by default on Eagle.  For the compilers, we will be using the GNU compiler suite (gcc and gfortran).  These can be accessed on Eagle by loading the appropriate module.  In theory, this should work with any version of the GNU compilers but we used version 8.2.0.  These can be loaded by typing `module load gcc/8.2.0`.

### Metis (Optional)

Metis helps the HSL solvers perform better.  Therefore it is recommended that you also install or build the Metis library.  If you do want to install Metis, it must be done before compiling the HSL library.

On Eagle, the easiest way to install Metis is to use anaconda:
```
conda install -c conda-forge metis
```
**Note**: The conda executable is accessed on Eagle by loading the conda module: `module load conda`

**Note**: Anaconda packages sometimes have issues when they come from different channels.  We tend to pull everything from conda-forge hence the channel choice above.

To make the metis header and dynamic library easily accessible to the HSL, MUMPS and IPOPT libraries, we will put symbolic links in the `${MYINC}` and `${MYLIB}` directories.  Do this by doing the following:
```
cd ${MYINC}
ln -s ${METIS_HEADER}/metis.h
cd ${MYLIB}
ln -s ${METIS_LIBRARY}/libmetis.so
```
This has a couple of advantages.  First, the coinbrew build will automatically add the `${MYLIB}` directory to the rpath of all constructed libraries and executables.  This means that we don't need to add `${MYLIB}` to the LD_LIBRARY_PATH.  The other advantage is that anaconda puts all the environments libraries and include files in the same directories with `libmetis.so` and `metis.h`.  Many of these libraries overlap with those used by HSL, Mumps and Ipopt but are not necessarily the same versions.  Loading a different version of a library than those compiled against can cause unexpected behavior.

### Building HSL and Ipopt

We will use COIN-OR's [coinbrew](https://github.com/coin-or/coinbrew) repo to build Ipopt along with the dependencies ASL, HSL and Mumps libraries.

1. `module load gcc/8.2.0 mkl conda`
2. clone (or download) the [coinbrew](https://github.com/coin-or/coinbrew) repo
3. cd into the directory
4. `./coinbrew fetch Ipopt:stable/3.13`
    * this fetches the branch `stable/3.13` of the Ipopt repository as well as the dependencies COIN-OR repositories `ThirdParty-ASL`, `ThirdParty-HSL` and `ThirdParty-Mumps`
5. `cd ThirdParty/HSL`
6. copy the HSL source code to the current directory and unpack it
7. create a link called `coinhsl` that points to the HSL source code (or rename the directory)
8. `cd ../..`
9. Configure and build everything:
`./coinbrew build Ipopt --disable-java --prefix="${MYAPPS}" --with-metis-cflags="-I${MYINC}" --with-metis-lflags="-L${MYLIB} -lmetis" --with-lapack-lflags="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl" --with-lapack-cflags="-m64 -I${MKLROOT}/include" ADD_CFLAGS="-march=skylake-avx512" ADD_FCFLAGS="-march=skylake-avx512" ADD_FFLAGS="-march=skylake-avx512"`
    * `build Ipopt` tells `coinbrew` to configure and build Ipopt and its dependencies
    * `--disable-java` says to build Ipopt without the java interface
    * `--prefix` says to install the library in "${MYAPPS}"
    * `--with-metis-cflags` gives the compiler the location of the metis header "metis.h"
    * `--with-metis-lflags` gives the linker the location and name of the metis library
    * `--with-lapack-lflags` gives the location of LAPACK and BLAS libraries as well as the needed linker lines.  Here we are using Intel's single dynamic library interface (google "mkl single dynamic library" for more details on this).
    * `ADD_CFLAGS`, `ADD_FCFLAGS` and `ADD_FFLAGS` say to use those extra flags when compiling C and fortran code, respectively. Specifically, `-march=skylake-avx512` tells the compiler to optimize code for the skylake CPUs on Eagle which is [recommended for perfomance reasons](https://www.nrel.gov/hpc/eagle-software-libraries-mkl.html).

**Note**: When linking with MKL libraries, Intel's [link line advisor](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html) is extremely helpful.

**Note**: When compiling Julia with MKL libraries, the single dynamic library interface is used to link against.  This is why we are also using that linking method.  Using a different linking method will cause unusual behaviors when using Ipopt with Julia (e.g. through JuMP).

### Using Custom Ipopt with JuMP

To use our custom installation of Ipopt with Ipopt.jl, we do the following:

1. Open the Julia REPL and activate an enviroment that has Ipopt installed
2. `ENV["JULIA_IPOPT_LIBRARY_PATH"] = ENV["MYLIB"]; ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = ENV["MYBIN"]`
    * tell Julia and Ipopt.jl the location of our Ipopt library and exectuble
3. `using Pkg; Pkg.build("Ipopt");`
    * rebuild Ipopt.jl with the above enviroment variables set to pick up the new library and executable
4. `using Ipopt; println(Ipopt.libipopt_path)`
    * print the path Ipopt.jl has stored for `libipopt.so`. This should be the location of your compiled version.

# Mac OS X

The following provides detailed instructions for compiling HSL library for Ipopt on Mac OS X

### Environment

We will make use of the following environment variables.  I have given the values that I used.

```
# Location of metis.h
export METIS_HEADER=/usr/local/include
# Location of metis library
export METIS_LIBRARY=/usr/local/lib

# Directory for keeping source code and build products
export MYAPPS=${HOME}/MyApps
# Location of static and dynamic libraries
export MYLIB=${MYAPPS}/lib

# Make dynamic libraries visible to the OS
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MYLIB} # Linux
export DYLD_LIBRARY_PATH=${DYLID_LIBRARY_PATH}:${MYLIB} # Mac OS X
```

These can be added to the .bash_profile file (or equivalent for other shells).  Remember after adding these to source .bash_profile (or equivalent) or to open a new terminal and do all building there.

### Prerequisites

As noted in the [Ipopt install instructions](https://coin-or.github.io/Ipopt/INSTALL.html), we will need `pkg-config` and a C, C++ and Fortran compiler.  `pkg-config` is available with either anaconda or homebrew in the expected manner (i.e. `conda install pkg-config` or `brew install pkg-config`).  For the compilers, it is recommended to use the GNU compiler suite (gcc, g++ and gfortran) which is what we will be doing here.

To avoid overwriting access to the clang compilers for other programs (which on Mac are aliased to gcc and g++ assuming you have installed the developer tools), we will install the compilers with homebrew.  This is done by `brew install gcc`.  As of this writing, this will install version 9 of the GNU compilers (version 8 can be installed with `brew install gcc@8`).  We will assume that version 9 is being used.

### Metis (Optional)

Metis helps the HSL solvers perform better.  Therefore it is recommended that you also install or build the Metis library.  If you do want to install Metis, it must be done before compiling the HSL library.

For mac, the easiest way to install Metis is to use Homebrew.  This is done with
```
brew install metis
```

Metis is also available from Anaconda.

### HSL

We will use a COIN-OR repo for building the library. It is [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL).

1. clone (or download) the [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) repo
2. cd into the directory
3. copy to current directory and unpack the HSL source code
4. create a link called "coinhsl" that points to the HSL source code (or rename the directory)
5. `mkdir build && cd build` -- not necessary but recommended
6. `../configure F77=gfortran-9 FC=gfortran-9 CC=gcc-9 --prefix="${MYAPPS}" --with-metis --with-metis-lflags="-L${METIS_LIBRARY} -lmetis" --with-metis-cflags="-I${METIS_HEADER}"` -- setup for building; "CC=", "F77=" and "FC=" say to use "gcc-9" as the C compiler and "gfortran-9" for the fortran and fortran 77 compilers, "--with-metis" says to use Metis in the build, "--with-metis-lflags" gives the linker the location and name of the metis library, "--with-metis-cflags" gives the compiler the location of the metis header "metis.h" and "--prefix=" says to install the library in "${MYAPPS}".
7. `make && make install`

### Using with Ipopt in JuMP

Ipopt has a feature called the linear solver loader (read about it [here](https://coin-or.github.io/Ipopt/INSTALL.html#LINEARSOLVERLOADER)) which allows for loading linear solvers from a dynamic library at run time.  We will use this feature to use the HSL solvers.

The only thing you have to do is to make the HSL dynamic library findable.  This is done by adding the directory containing the HSL library to the environment variable "DYLD_LIBRARY_PATH".  This was done in the "Environment" section.  So we should be good to go.  To use the new linear solvers just use the `linear_solver="<solver>"` argument to `Ipopt.Optimizer`.

The following julia code is useful for testing the HSL linear solvers are working

```
using JuMP, Ipopt

m = JuMP.Model(()->Ipopt.Optimizer(linear_solver="ma97"))
@variable(m, x)
@objective(m, Min, x^2)
JuMP.optimize!(m)
```

**Note**: The Ipopt build that comes with Ipopt.jl seems to expect the HSL library to have the name "libhsl.dylib". The repo "ThirdParty-HSL" builds the library "libcoinhsl.dylib".  The simplest fix is to go to `${MYLIB}` and put in a link that points "libhsl.dylib" to "libcoinhsl.dylib".

