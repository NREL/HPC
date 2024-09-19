---
layout: main
title: IDAES Solvers
parent: Applications

---

# IDAES Solvers

*Institute for Design of Advanced Energy Systems (IDAES) Solvers are a collection of pre-compiled optimizer binaries with efficient linear algebra solvers that enable solving a variety of MINLP problems.*

Available optimizers include:

1. Bonmin
2. CBC
3. CLP
4. Couenne
5. IPOPT

## Available Modules

| Kestrel                              |
|:------------------------------------:|
| idaes_solvers/3.4.0-cray-libsci      |
| idaes_solvers/3.4.0-intel-oneapi-mkl |
| idaes_solvers/3.4.0-netlib-lapack    |

### v3.4.0

IDAES Solvers v3.4.0 contains the following optimizer versions

| Optimizer   | Version |
|:-----------:|:-------:|
| Bonmin      | 1.8.8   |
| CBC         | 2.10.10 |
| CLP         | 1.17.8  |
| Couenne     | 0.5.8   |
| IPOPT + HSL | 3.13.2  |

!!! note
    IPOPT is available with performant HSL MA27 and MA57 linear solvers. These have been shown to perform better than the default MUMPS solver for a variety of renewable energy optimization problems. Please see documentation [here](ipopt.md#ipopt--hsl)

## Usage

Users can run any of the IDAES solvers simply by loading the appropriate module, e.g.,

```bash
module load idaes_solvers/3.4.0-intel-oneapi-mkl
```

!!! note
    IDAES solvers dependent on LAPACK require users to load the current `netlib-lapack` module. Please see [here](ipopt.md#usage) for explicit instructions.
