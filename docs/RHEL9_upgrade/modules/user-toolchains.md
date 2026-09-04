# Selecting a User-Provided Toolchain

On Kestrel RHEL 9, you can build your own software environment by selecting individual compiler toolchains. This is separate from and incompatible with the CPE stack.

## What are User-Provided Toolchains?

User-provided toolchains are individual compilers and their associated libraries that you select à la carte:

- **gcc**: GNU Compiler Collection (gcc, g++, gfortran)
- **oneapi**: Intel oneAPI suite (icx, icpx, ifort)
- **llvm**: LLVM/Clang compiler stack

When you load a compiler base module, Lmod reveals only software built for that specific toolchain. This ensures compatibility.

## Starting Fresh

Always begin with a clean slate:

```bash
module reset
```

This restores your environment to the login defaults (Core modules + DefApps) and avoids conflicts.

## Loading a GCC Toolchain

### Discover available GCC versions

```bash
module spider gcc
```

### Load a specific GCC version

```bash
module load gcc/14.2.0
```

### View available software for this toolchain

Once GCC is loaded, see what libraries and tools are available:

```bash
module avail
```

Example output:

```bash
----------- [ gcc/14.2.0 ] -----------
   adios/1.13.1              fftw/3.3.10           openblas/0.3.32    
   darshan-util/3.4.7 (D)    gsl/2.8     (D)       netlib-lapack/3.12.1 
   zlib/1.3.2
```

### Accessing GPU-enabled software compiled with GCC

To access prebuilt GPU-aware modules (and dependencies) that were compiled with `gcc/14.2.0` and `cuda/13.2`, explicitly load these modules first:

```bash
module load gcc/14.2.0 cuda/13.2
```

Example output:

```bash
-------------------------------------------------------- [ gcc/14.2.0, cuda/13.2 ] ---------------------------------------------------------
   hwloc/2.12.2 (D)    mpich/5.0.0 (D)    nccl/v2.27.7-1 (D)    nccl/v2.29.2-1    openmpi/5.0.3 (D)    shs-libfabric/shs-12.0.2
```


## Loading a oneAPI Toolchain

### Discover available oneAPI versions

```bash
module spider oneapi
```

### Load a specific oneAPI version

```bash
module load oneapi/2025.3.1
```

### View available software for this toolchain

```bash
module avail
```

Example output:

```bash
----------- [ oneapi/2025.3.1 ] -----------
   fftw/3.3.10                          intel-oneapi-mkl/2025.3.1   (D)
   intel-oneapi-advisor/2025.4.1        intel-oneapi-tbb/2021.10.0
   intel-oneapi-dal/2024.0.0     (D)    intel-oneapi-vtune/2025.8.1
   intel-oneapi-inspector/2024.1.0      mpich/5.0.0
```

## Discovering Software Variants

Use `module spider` to find all available versions across all toolchains:

```bash
module spider hdf5
```

This shows all variants. To see dependencies for a specific version:

```bash
module spider hdf5/1.14.5
```



## Loading Additional Software

After selecting a compiler, load compatible libraries and tools:

```bash
module load gcc/14.2.0
module load fftw/3.3.10
module load openblas/0.3.32
module load gsl/2.8
```

Core modules are always available regardless of toolchain:

```bash
module load cmake/3.31.11
module load git/2.52.0
module load gdb/16.2
```

## Switching Toolchains

To switch from one toolchain to another, start fresh:

```bash
module reset
module load oneapi/2025.3.1
```

Or use `module swap`:

```bash
module swap gcc/14.2.0 oneapi/2025.3.1
```

⚠️ **Do not mix toolchains.** Swap cleanly; don't load both simultaneously.

## ⚠️ NLR Toolchains vs CPE

NLR-provided toolchains and the Cray Programming Environment (CPE) are **mutually exclusive**:

```bash
# DO NOT DO THIS
module load gcc/14.2.0
module load cpe-stack/25.03    # Conflict!
```

**Choose one workflow:**

- **User toolchains:** `gcc`, `oneapi`, or `llvm` (individual selection)
- **CPE:** `cpe-stack/25.03` (complete pre-built environment)

See [Cray Programming Environment](cpe.md) for the CPE workflow.

## Recommended Practices

1. Start each session with `module reset`
2. Load only one compiler at a time
3. Use `module avail` to see what's compatible before loading
4. Use `module spider` to search globally
5. Keep session notes if switching between toolchains frequently

## Quick Reference

| Task | Command |
| --- | --- |
| See available gcc versions | `module spider gcc` |
| Load gcc | `module load gcc/14.2.0` |
| See available oneapi versions | `module spider oneapi` |
| Load oneapi | `module load oneapi/2025.3.1` |
| View compatible packages | `module avail` |
| Find a package across all toolchains | `module spider fftw` |
| Reset environment | `module reset` |



## Recommended practice

- Keep to one software stack per shell session.
- Prefer `module spider` for complete discovery.
- Use `module reset` before switching to a different toolchain.


This workflow reduces the risk of mixing incompatible compiler stacks.
