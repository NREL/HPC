---
layout: main
title: IDAES Solvers
parent: Applications

---

# IDAES Solvers

*Institute for Design of Advanced Energy Systems (IDAES) Solvers are a collection of pre-compiled optimizer binaries with efficient linear algebra solvers that enable solving a variety of MINLP problems.*

Available optimizers include:

1. [Bonmin](#bonmin)
2. [CBC](#cbc)
3. [CLP](#clp)
4. [Couenne](#couenne)
5. [IPOPT + HSL](#ipopt-hsl)

## Available Modules

!!! info
    IDAES solvers are currently not available on GPU compute nodes.

| Kestrel (CPU nodes)                  |
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
    IPOPT is available with performant HSL MA27, MA57, and MA97 linear solvers. These have been shown to perform better than the default MUMPS solver for a variety of renewable energy optimization problems. Please see documentation [here](ipopt.md#ipopt--hsl).

### Usage

Users can run any of the IDAES solvers simply by loading the appropriate module, e.g.,

```bash
module load idaes_solvers/3.4.0-cray-libsci # OR 
module load idaes_solvers/3.4.0-netlib-lapack # OR
module load idaes_solvers/3.4.0-intel-oneapi-mkl
```

## Bonmin

Bonmin (Basic Open-source Nonlinear Mixed Integer) is an open source solver that leverages CBC and IPOPT to solve general mixed integer nonlinear programs (MINLP). 
Please refer to the Bonmin [documentation here](https://coin-or.github.io/Bonmin/)

## CBC

COIN-OR Branch and Cut (CBC) solver is an opensource optimizer for solving mixed integer programs (MIP). Please refer to the [documentation here](https://coin-or.github.io/Cbc/intro.html) for more details.

## CLP

COIN-OR Linear Program (CLP) is an open-source solver for solving linear programs. Please refer to the [documentaion here](https://coin-or.github.io/Clp/) for further details. 

## Couenne

Convex Over and Under Envelopes for Nonlinear Estimation (Couenne) is an open-source mixed integer nonlinear programming (MINLP) global optimization solver. Please visit the [following website](https://github.com/coin-or/Couenne) for more details regarding the solver.

## IPOPT + HSL

Interior Point Optimizer (IPOPT) is an open-source nonlinear optimizer.
[Harwell Subroutine Library (HSL)](https://licences.stfc.ac.uk/product/coin-hsl) is a collection of efficient linear solvers used by IPOPT.
HSL solvers have been demonstrated to be more performant than the default [MUMPS](https://mumps-solver.org/index.php) (Multifrontal Massively Parallel sparse direct Solver) solver that comes with IPOPT, and are highly recommended.

IPOPT that is distributed as part of IDAES solvers comes pre-compiled with 3 HSL solvers:

1. **MA27** is a serial linear solver suitable for small problems
2. **MA57** has threaded BLAS operations and is suitable for small to medium-sized problems.
3. **MA97** is a parallel direct linear solver for sparse symmetric systems. It is more suitable for medium and large problem sizes. Users will may see worse performance on small problems when compared to MA27 and MA57.

All three solvers produce repeatable answers unlike their sibling MA86.

!!! info
    For additional details regarding IPOPT on Kestrel, e.g., building a custom version, please visit [here](./ipopt.md). Please click here for additional details regarding [HSL](../Development/Libraries/hsl.md) solvers on Kestrel.