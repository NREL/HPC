# Quick Start

## Accessing RHEL9

All users are encouraged to test their workflows on a set of nodes that are already upgraded to RHEL9 to prepare for the system-wide transition. To do so, you must first be connected to a [dedicated RHEL9 login node](./access.md). Please reach out to [HPC-Help@nlr.gov](mailto:hpc-help@nlr.gov) with any issues or questions.

## Frequently Asked Questions (FAQ)

Please see our dedicated [page](./knownissues.md#frequently-asked-questions-faq) for FAQs about RHEL9 on Kestrel.

## Using Legacy (RHEL8) Software

Kestrel users can still use old (RHEL8) software modules **at their own risk** and with the understanding that those modules **will no longer be maintained** by support staff after the OS upgrade. Refer to the dedicated [Legacy Software](./modules/old_software.md) page for more information.

## Software modules on RHEL9 

!!! Note
    This subsection is targeted towards users who manage their own software environments and need access to specific modules for compilers, MPI, scientific libraries, etc. The new module system still allows for end-use applications to be loaded directly (e.g., `module load vasp`). If you primarily run such end-use applications, you should not notice a major change in your experience using Kestrel.

Kestrel's module system design is [significantly changing](./modules/index.md) with the RHEL9 upgrade. In general, to use the [Cray Programming Environments (CPE)](./modules/cpe.md), you must first run `module load cpe-stack` for the CPE modules to become available. For environments outside of CPE (e.g., [a user-defined one](./modules/user-toolchains.md) built with GNU), you must load the compiler first as a "base module" to reveal software built for that specific toolchain.

!!! Warning
    Mixing CPE and non-CPE environments in the same job or build script is not recommended.

### RHEL9 Module "Cheat Sheet"

Please refer to the non-exhaustive translation table below to note how to load the equivalent module(s) between RHEL8 and RHEL9 for common module operations.

| RHEL8 module(s)                    | RHEL9 equivalent                    | RHEL9 environment type |
| :--                                | :--                                 | :--                    |
| `vasp`                             | `vasp`                              | End-use application    |
| `PrgEnv-intel intel-oneapi-mpi`    | `cpe-stack oneapi intel-oneapi-mpi` | CPE                    |
| `PrgEnv-cray`                      | `cpe-stack PrgEnv-cray`             | CPE                    |
| `cray-mpich`                       | `cpe-stack PrgEnv-cray`             | CPE                    |
| `gcc-stdalone`                     | `gcc`                               | User-defined           |
| `conda`                            | `miniforge3`                        | Python                 |
