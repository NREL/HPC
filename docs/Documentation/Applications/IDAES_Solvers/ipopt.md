# IPOPT + HSL

Interior Point Optimizer (IPOPT) is an open-source nonlinear optimizer.
[Harwell Subroutine Library (HSL)](https://licences.stfc.ac.uk/product/coin-hsl) is a collection of efficient linear solvers used by IPOPT.
HSL solvers have been demonstrated to be more performant than the default [MUMPS](https://mumps-solver.org/index.php) (Multifrontal Massively Parallel sparse direct Solver) solver that comes with IPOPT, and are highly recommended.

IPOPT that is distributed as part of IDAES solvers comes pre-compiled with 2 HSL solvers:

1. MA27 is a serial linear solver suitable for small problems
2. MA57 has threaded BLAS operations and is suitable for small to medium-sized problems.

Both these solvers produce repeatable answers.

## Usage

Users can use IPOPT by simply loading the IDAES solver module, e.g., 

```bash
module load netlib-lapack
module load idaes_solvers/3.4.0-netlib-lapack 
```