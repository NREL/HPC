---
layout: default
title: Gurobi
---

*Gurobi Optimizer is a suite of solvers for mathematical programming.*

For documentation, forums, and FAQs, see the [Gurobi
website](https://www.gurobi.com/products/gurobi-optimizer/).

Gurobi includes a linear programming solver (LP), quadratic programming solver
(QP), quadratically constrained programming solver (QCP), mixed-integer linear
programming solver (MILP), mixed-integer quadratic programming solver (MIQP),
and a mixed-integer quadratically constrained programming solver (MIQCP).

NREL hosts 6 general use (including commercial) and 18 academic/government
standalone Gurobi licenses. The Gurobi interactive shell will run by typing
"`gurobi.sh`". Gurobi can also be interfaced with C/C++/Java/MATLAB/R codes by
linking with the Gurobi libraries.

For details on Gurobi programming, see the [Gurobi Resource
Center](https://www.gurobi.com/resource-center/) and [Gurobi
documentation](https://www.gurobi.com/documentation/).

## Gurobi and MATLAB

To use the Gurobi solver with MATLAB, make sure you have the Gurobi and MATLAB
environment modules loaded, then issue the following two commands from the
MATLAB prompt or your script:

```
>> grb = getenv('GRB_MATLAB_PATH')
>> path(path,grb)
```

## Gurobi and General Algebraic Modeling System

The General Algebraic Modeling System (GAMS) is a high-level modeling system for
mathematical programming and optimization. The GAMS package installed at NREL
includes Gurobi solvers. For more information, see [using GAMS](gams.md).

Note that the Gurobi license for this interface is separate from the standalone
Gurobi license, and supports far more instances.

**Important: When using the Gurobi solver in GAMS, the user should NOT load the
Gurobi module. Simply using "module load gams" will be enough to load the
required Gurobi components and access rights.**
