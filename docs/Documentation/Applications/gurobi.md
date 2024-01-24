# Using the Gurobi Optimizer Solvers

*Gurobi Optimizer is a suite of solvers for mathematical programming.*

For documentation, forums, and FAQs, see the [Gurobi
website](https://www.gurobi.com/products/gurobi-optimizer/).

Gurobi includes a linear programming solver (LP), quadratic programming solver
(QP), quadratically constrained programming solver (QCP), mixed-integer linear
programming solver (MILP), mixed-integer quadratic programming solver (MIQP),
and a mixed-integer quadratically constrained programming solver (MIQCP).

Gurobi is available on multiple systems, which host 6 general use (including
commercial) and 18 academic/government standalone Gurobi licenses. After logging
onto the appropriate cluster, load the default Gurobi module using 
`module load gurobi`.  The Gurobi interactive shell is run by typing 
"`gurobi.sh`". Gurobi can also be interfaced with C/C++/Java/MATLAB/R codes by 
linking with the Gurobi libraries.

For details on Gurobi programming, see the [Gurobi Resource
Center](https://www.gurobi.com/resource-center/) and [Gurobi
documentation](https://www.gurobi.com/documentation/).

## Available Modules

| Kestrel         | Eagle           | Swift           |
|:---------------:|:---------------:|:---------------:|
| gurobi/11.0.0   |||
| gurobi/10.0.2   | gurobi/10.0.2   ||
| gurobi/10.0.1   | gurobi/10.0.1   ||
| gurobi/9.5.1    | gurobi/9.5.1    | gurobi/9.5.1    |


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

!!! tip "Important"
    When using the Gurobi solver in GAMS, the user should NOT load the
    Gurobi module. Simply using "module load gams" will be enough to load the
    required Gurobi components and access rights.
