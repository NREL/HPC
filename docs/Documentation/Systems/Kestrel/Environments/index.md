# Kestrel Programming Environments Overview

## Definitions

**Toolchain**: a combination of a compiler and an mpi library. Sometimes associated scientific libraries (scalapack, blas, etc.) or bundles of scientific libraries (MKL, libsci, etc.) are considered part of the toolchain.

**Environment**: a set of modules, including a toolchain. A "build environment" refers to the set of modules (including compiler and MPI library) used to compile a code. A "run-time environment" is the set of modules used to execute a code. The two typically, but not always, match.

## Environments

Kestrel runs Red Hat Enterprise Linux 9, and its module system is [Lmod](../../../Environment/lmod.md) 8.7.37, a Lua-based, hierarchical module system. At login, a minimal default set of modules is loaded (`Core/26.05` and `DefApps`).

There are three distinct environment types on Kestrel. **Do not mix them in the same session** — always start with `module reset` before switching.

1. **Cray Programming Environment (CPE)** — a complete, pre-validated software stack provided by HPE Cray, activated with `module load cpe-stack/25.03` followed by a `PrgEnv-*` module.
2. **NLR-provided toolchains** — individual compiler toolchains (`gcc`, `oneapi`, `llvm`) that you load à la carte, then layer MPI and libraries on top.
3. **Miniforge / Containers** — `miniforge3` for user-managed Conda/Mamba Python environments, and `apptainer` for running pre-built container images. These are independent of the compiler-based workflows above.

### 1. Cray Programming Environment (CPE)

CPE bundles multiple compilers (Cray, GNU, Intel, AOCC, NVIDIA), a pre-integrated MPI (Cray-MPICH), and pre-tested libraries and tools. This is the closest analog to the "PrgEnv-" environments from Kestrel's RHEL8 days, but it must now be explicitly activated:

```bash
module reset
module load cpe-stack/25.03
module load PrgEnv-gnu      # or PrgEnv-cray, PrgEnv-intel, PrgEnv-aocc, PrgEnv-nvidia
```

Each `PrgEnv-*` includes a compiler suite, Cray-MPICH, Cray LibSci (a drop-in alternative to MKL), and additional communication/network libraries. Switch between compilers within CPE using `module swap PrgEnv-gnu PrgEnv-intel`, which preserves the rest of the CPE stack.

Use the Cray compiler wrappers `cc`, `CC`, and `ftn` — they automatically invoke the correct underlying compiler and MPI for whichever `PrgEnv-*` is loaded.

### 2. NLR-Provided Toolchains

NLR-provided toolchains let you select a compiler and its associated libraries individually, similar to the old Eagle-style module experience. Loading a compiler base module reveals only the software built for that specific toolchain:

```bash
module reset
module load gcc/14.2.0        # or oneapi/2025.3.1, or llvm
module avail
```

MPI (OpenMPI, MPICH, or Intel oneAPI MPI) is then loaded on top of the compiler — see [MPI](../../../Development/Programming_Models/mpi.md) for details. GPU-enabled software is revealed by also loading a `cuda` module, e.g. `module load gcc/14.2.0 cuda/13.2`.

### 3. Miniforge and Containers

For user-managed Python/R/data-science stacks, use `module load miniforge3/26.1.1-3` and manage packages with `conda`/`mamba`. For portable, pre-built software stacks, use `module load apptainer/1.4.1-runonly` and run images with `apptainer exec`.

## A Note on OpenMPI

On Kestrel, Cray-MPICH (via CPE) generally offers the best performance and stability for multi-node jobs, since it is tuned for Kestrel's Slingshot network. OpenMPI is available via the NLR toolchain workflow, but `openmpi5` currently only works with `mpirun` (not `srun`), because Slurm is configured without PMIx support — see the [MPI](../../../Development/Programming_Models/mpi.md) page for the current workaround. If you have trouble with OpenMPI, please reach out to hpc-help@nlr.gov.

## Summary of Available Compiler Environments

* **CPE** compilers/MPI are activated with `cpe-stack` + `PrgEnv-*`.
* **NLR toolchains** are activated by loading the compiler module directly (`gcc`, `oneapi`, `llvm`).

### GNU

| Workflow | Compiler Module | MPI module |Language|Wrapper|Compiler|MPI|
|-------|---------------|------------|--------|-------|--------|---|
| CPE (`PrgEnv-gnu`) | gcc (Cray)    | cray-mpich | Fortran| ftn   |gfortran| Cray MPICH |
| CPE (`PrgEnv-gnu`) | gcc (Cray)    | cray-mpich | C      | cc    | gcc    | Cray MPICH |
| CPE (`PrgEnv-gnu`) | gcc (Cray)    | cray-mpich | C++    | CC    | g++    | Cray MPICH |
| NLR toolchain   | gcc/14.2.0   | openmpi/5.0.5 | Fortran| mpifort| gfortran| Open MPI|
| NLR toolchain   | gcc/14.2.0   | openmpi/5.0.5 | C      | mpicc | gcc | Open MPI|
| NLR toolchain   | gcc/14.2.0   | openmpi/5.0.5 | C++    | mpic++ | g++ | Open MPI|

### Cray 

| Workflow | Compiler Module | MPI module|Language|Wrapper|Compiler|MPI|
|-------|---------------|-----------|--------|-------|--------|---|
| CPE (`PrgEnv-cray`)  | cce (Cray)   |cray-mpich  |Fortran| ftn   | crayftn| Cray MPICH |
| CPE (`PrgEnv-cray`)  | cce (Cray)   |cray-mpich  |C      | cc    | craycc | Cray MPICH |
| CPE (`PrgEnv-cray`)  | cce (Cray)   |cray-mpich  |C++    | CC    | crayCC | Cray MPICH |

### Intel

| Workflow | Compiler Module | MPI Module|Language|Wrapper|Compiler|MPI|
|-------|---------------|------|--------|-------|--------|---|
| CPE (`PrgEnv-intel`) | intel (Cray)  | cray-mpich    | Fortran| ftn   | ifx  | Cray MPICH |
| CPE (`PrgEnv-intel`) | intel (Cray)  | cray-mpich    | C      | cc    | icx    | Cray MPICH |
| CPE (`PrgEnv-intel`) | intel (Cray)  | cray-mpich    | C++    | CC    | icpx    | Cray MPICH |
| NLR toolchain   | oneapi/2025.3.1 |intel-oneapi-mpi/2021.17.2  | Fortran| mpiifort | ifx  | Intel oneAPI MPI|
| NLR toolchain   | oneapi/2025.3.1 |intel-oneapi-mpi/2021.17.2  | C      | mpiicx   | icx    | Intel oneAPI MPI|
| NLR toolchain   | oneapi/2025.3.1 |intel-oneapi-mpi/2021.17.2  | C++    | mpiicpx  | icpx   | Intel oneAPI MPI|

!!! note
    The Intel Classic compilers (`icc`, `ifort`) have been fully retired in favor of the oneAPI compilers (`icx`, `icpx`, `ifx`). See [MPI](../../../Development/Programming_Models/mpi.md) and the [oneAPI Toolchain tutorial](Toolchains/intel.md) for details.

Note: 

The `Cray MPICH` used for each different `PrgEnv-` is pointing to a different instance of MPICH; check `module show cray-mpich` after loading a given `PrgEnv-*` to see the exact path in use.

## PrgEnv- Programming Environments (CPE)

### Introduction

These environments come packaged with:

1. A compiler, which corresponds to the name of the environment. E.g., `PrgEnv-intel` uses Intel compilers
2. Cray MPICH
3. Cray LibSci, which can be used in place of MKL
4. Additional communication and network libraries

To use a `PrgEnv-*`, you must first load `cpe-stack/25.03`. We can swap between programming environments using the `module swap` command. For example, if `PrgEnv-gnu` is loaded but we want to use `PrgEnv-cray` instead, we can `module swap PrgEnv-gnu PrgEnv-cray`.

### What is a PrgEnv module doing?

PrgEnv modules can seem a bit mysterious. We can check out the inner workings of a PrgEnv module with the `module show` command. For example, for `PrgEnv-gnu` we can:

`module show PrgEnv-gnu`

This tells us that PrgEnv-gnu conflicts with all other PrgEnvs. The modulefile sets some environment variables (the `setenv` lines), and loads the modules associated with the programming environment.

We can use the `module whatis` command to give us a brief summary of a module. For example, the command:

`module whatis craype`

outputs:

`craype               : Setup for Cray PE driver set and targeting modules`

We mentioned previously that the different PrgEnvs use different locations for Cray-MPICH. We can see this by using `module show cray-mpich` in each different PrgEnv, and examining (for example) the `CRAY_LD_LIBRARY_PATH` environment variable. 


### Compiling inside a PrgEnv: ftn, cc, and CC

`ftn`, `cc`, and `CC` are the Cray compiler wrappers for Fortran, C, and C++, respectively, which are part of the `craype` module. When a particular `PrgEnv-` programming environment is loaded, these wrappers will make use of the corresponding compiler. For example, if we load PrgEnv-gnu with:

```
module swap PrgEnv-cray PrgEnv-gnu
```

we would expect `ftn` to wrap around gfortran, the GNU fortran compiler. We can test this with:

`ftn --version`

Note1: In contrast with mpich, the location of the wrappers `cc`, `CC` and `ftn` is always the same regardless of the loaded PrgEnv (part of the `craype` module).

Note2: `cc`, `CC` and `ftn` are also wrappers around their mpi couterparts. For mpi codes, the wrappers call the necessary mpi compilers depending on which PrgEnv is loaded. 

Note3: When changing between PrgEnvs, it is better to use `module swap [current prgenv] [new prgenv]` instead of `module reset; module load [new prgenv]` due to the way the environments set some environment variables.

Note: you can still directly access the underlying compiler. For example, if we're using PrgEnv-gnu (so our compilers are the GCC compilers), we can use `ftn`, or we can use `gfortran` or `mpifort` directly. It is considered best practice to use the Cray wrappers (`ftn`, `cc`, `CC`) on a Cray machine like Kestrel.

In fact, the use of `mpifort` can be quite confusing. Inside the PrgEnv-gnu environment, `mpifort` wraps around Cray MPICH, not OpenMPI. If instead you load the NLR `gcc` toolchain and then `openmpi`, `mpifort` will wrap around OpenMPI. Using the Cray wrappers (`ftn`, `cc`, `CC`) inside CPE helps avoid this confusion.

### Submitting a job within a PrgEnv

Submitting a Slurm job using a PrgEnv environment is no different than how you would normally submit a job. In your slurm script, below the #SBATCH directives, include:

```
module reset
module load cpe-stack/25.03
module load PrgEnv-gnu   # or whichever PrgEnv you need
```

Depending on the software you're trying to run, you may need to load additional modules like `cray-hdf5` or `cray-fftw` (these are bundled with `cpe-stack`).

## NLR-Provided Toolchains

NLR-provided toolchains function similarly to the old Eagle-style modules, where it is up to the user to load all necessary modules to build and run their applications. These are not CPE and should not be combined with `cpe-stack`/`PrgEnv-*` in the same session.

```bash
module reset
module load gcc/14.2.0        # or oneapi/2025.3.1, or llvm
module avail
```

The modules are optimized for Kestrel architecture and updated periodically. If there is a module you need but is not available, email hpc-help@nlr.gov.

### MPI-Enabled Software

MPI-enabled software modules use a `-mpi` suffix and, like compiler-dependent modules, are not visible until the relevant MPI implementation is loaded (`openmpi`, `mpich`, or `intel-oneapi-mpi`). See [MPI](../../../Development/Programming_Models/mpi.md) for details.

### cray-mpich-abi

For binaries dynamically built with an MPICH-based MPI (NLR toolchain, e.g. `intel-oneapi-mpi`), `cray-mpich-abi` can be used at runtime to leverage Cray-MPICH's optimization for Kestrel. Check `module spider cray-mpich-abi` to confirm current availability, since this is an advanced binary-compatibility feature.

## Discovering Modules

Use `module avail` to see what's compatible with your currently loaded modules, and `module spider <name>` to search globally (including modules hidden by your current environment) and see what prerequisites are needed:

```bash
module avail
module spider fftw
module spider fftw/3.3.10
```

See [Module Commands](../../../../RHEL9_upgrade/modules/ml_commands.md) for a complete command reference, and the [Module Translation Table](../../../../RHEL9_upgrade/quick_start.md#rhel8-rhel9-module-translation-table-cpu-side) if you're migrating scripts from the earlier RHEL8 module names.

