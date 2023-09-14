# mpi_version_check

The code check_mpi_version.f90 prints out what version of MPI you are running. The "load" files allow you to set up the desired environment on Kestrel.

1. load_PrgEnv_cray.sh: loads the default Cray "PrgEnv-cray" environment: Cray CCE compilers and Cray MPICH
2. load_PrgEnv_intel.sh: loads the default Cray "PrgEnv-intel" environment: Intel OneAPI compilers and Cray MPICH
3. load_NREL_intel.sh: loads the NREL-built Intel environment: Intel OneAPI compilers and Intel MPI
4. load_NREL_gcc_openmpi.sh: loads the NREL-built GNU/OpenMPI environment: GNU GCC compilers and Open MPI

To load a given environment, use the `source` command. E.g., for the GCC/OpenMPI environment, `source load_NREL_gcc_openmpi.sh`

Note: you can source these files for other work than just compiling this code. However, you may need to load a few additional modules for the two NREL-built environments. If you run into errors, try adding `module load cray-dsmml`, `module load libfabric`, and `module load craype-network-ofi`. If you are using a Cray `PrgEnv` you may additionally need to load libraries like `cray-fftw` and `cray-hdf5-parallel`. `cray-libsci` may be used in place of `mkl`. If using a `PrgEnv`, you may run into issues if you "mix and match" with NREL built modules (e.g. trying to use an NREL-built vrsion of hdf5)

Note: you can see how to load each environment by looking inside the corresponding "load" file.

Note: Make sure to include `--mpi=pmi2` in your `srun` for each section, or the code will not run properly.

**Environment**: An "environment" in this context refers to the compiler and MPI that you are building a code with.

## Tutorial

Log onto Kestrel: `ssh [your username]@kl1.hpc.nrel.gov`

Grab an interactive session: `salloc -N 1 -n 104 --time=1:00:00`

Clone the repo: `git clone https://github.nrel.gov/ohull/mpi_version_check.git`

Descend into the repo: `cd mpi_version_check`

### Environment 1: PrgEnv-cray

Load the PrgEnv-cray environment: `source load_PrgEnv_cray.sh`

Build the code: `ftn check_mpi_version.f90 -o version_PrgEnv_cray`

Run the code: `srun -N 1 -n 1 --mpi=pmi2 ./version_PrgEnv_cray`

Under the hood, what is `ftn` using? Run: `ftn --version` (Cray Fortran)

Note: this code does no checking about how many MPI tasks you are running with. If you ask for `-n 104`, it will print the version 104 times. So, just ask for one with `-n 1`.

You should see that the code ran with Cray MPICH.

What happens if you build with mpifort in this environment? Try it yourself with `mpifort check_mpi_version.f90 -o version_PrgEnv_cray_mpifort` and then `srun -N 1 -n 1 --mpi=pmi2 ./version_PrgEnv_cray_mpifort`. In this case, why does it still use Cray MPICH?

Note: it is best practice when using a `PrgEnv-` to use `ftn`, `cc`, or `CC` (for fortran, C, or C++ respectively) and NOT `mpifort`, `mpicc`, etc.

### Environment 2: PrgEnv-intel 

Repeat the exercise above, but with PrgEnv-intel:

Load the PrgEnv-cray environment: `source load_PrgEnv_intel.sh`

Build the code: `ftn check_mpi_version.f90 -o version_PrgEnv_intel`

Run the code: `srun -N 1 -n 1 --mpi=pmi2 ./version_PrgEnv_intel`

Under the hood, what is `ftn` using? Run: `ftn --version`

In this case, ftn tells us it is using Intel Fortran (ifort). Recall that `ftn` is a wrapper. In `PrgEnv-intel`, it is a wrapper around ifort, Intel's fortran compiler. In `PrgEnv-cray`, it is a wrapper around Cray Fortran instead.

After we run `srun -N 1 -n 1 --mpi=pmi2 ./version_PrgEnv_intel`, it should tell us that it is using Cray MPICH.
 
What happens if we try to use Intel's MPI compiler wrapper, `mpiifort`? Try `mpiifort --version`...Nothing happens, because we have not loaded Intel MPI into our environment.

### Environment 3: NREL Intel

Load the NREL Intel environment: `source load_NREL_intel.sh`

Build the code: `mpiifort check_mpi_version.f90 -o version_intel_intelmpi
Run the code: `srun -N 1 -n 1 --mpi=pmi2 ./version_intel_intelmpi

Note that we built using `mpiifort` instead of `ftn`. What happens if we try to use `ftn`? Run `ftn --version`. Nothing happens, because we are no longer using a Cray environment. In particular, `ftn` is contained inside the module `craype`.

After `srun`ing, we should see that our code is using Intel MPI.

### Environment 4: GCC/Open MPI

Load the NREL GCC/Open MPI environment: `source load_NREL_gcc_openmpi.sh`

Build the code: `mpifort check_mpi_version.f90 -o version_gcc_openmpi`

Run the code: `srun -N 1 -n 1 --mpi=pmi2 ./version_gcc_openmpi`

Now, we used `mpifort` to compile, and unlike in the Environment 1 section, it used Open MPI.
