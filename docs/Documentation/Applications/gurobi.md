# Gurobi

*Gurobi Optimizer is a suite of solvers for mathematical programming.*

!!! warning "License Request Required"
    Starting soon, Gurobi jobs will require explicit license requests in your job submission. 
    If you see the following warning message:
    
    ```
    WARNING: SLURM_JOB_LICENSES is not set.
    Please request a license with your job submission using '-L gurobi:numberoflicenses'.
    Gurobi will fail in the future if you do not add the above to your job submission.
    ```
    
    You must add the license request to your job submission using the `-L` flag. See the sbatch example [below](#requesting-gurobi-licenses-in-job-submissions).

Gurobi includes a linear programming solver (LP), quadratic programming solver
(QP), quadratically constrained programming solver (QCP), mixed-integer linear
programming solver (MILP), mixed-integer quadratic programming solver (MIQP),
and a mixed-integer quadratically constrained programming solver (MIQCP).

Gurobi is available on multiple systems. **There are 24 license tokens available for 
concurrent use** - 6 are for general use (including commercial) and 18 standalone license tokens
are for academic/government use. After logging
onto the appropriate cluster, load the default Gurobi module using 
`module load gurobi`.  The Gurobi interactive shell is run by typing 
"`gurobi.sh`". Gurobi can also be interfaced with C/C++/Java/MATLAB/R codes by 
linking with the Gurobi libraries.

For details on Gurobi programming, see the [Gurobi Resource
Center](https://www.gurobi.com/resource-center/) and [Gurobi
documentation](https://www.gurobi.com/documentation/).

## Available Modules

| Kestrel         | Swift           |
|:---------------:|:---------------:|
| gurobi/12.0.0   ||
| gurobi/11.0.2   ||
| gurobi/10.0.2   ||
| gurobi/10.0.1   ||
| gurobi/9.5.1    | gurobi/9.5.1    |

!!! tip
    You can check how many Gurobi licenses are available for use by running the following command
    after loading the Gurobi module
    ```bash
    gurobi_cl -t
    ```

## Requesting Gurobi Licenses in Job Submissions

When submitting jobs that use Gurobi, you need to request the appropriate number of licenses. Here's an example sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=gurobi_job
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -L gurobi:1

module load gurobi

# Your Gurobi commands here
gurobi.sh your_model.lp
```

The `-L gurobi:1` flag requests 1 Gurobi license token. Adjust the number based on your needs, keeping in mind that there are 24 license tokens available for concurrent use.


## Gurobi and MATLAB

To use the Gurobi solver with MATLAB, make sure you have the Gurobi and MATLAB
environment modules loaded, then issue the following two commands from the
MATLAB prompt or your script:

```
>> grb = getenv('GRB_MATLAB_PATH')
>> path(path,grb)
```

## Gurobi and General Algebraic Modeling System (GAMS)

The General Algebraic Modeling System (GAMS) is a high-level modeling system for
mathematical programming and optimization. The GAMS package installed at NREL
includes Gurobi solvers. For more information, see [using GAMS](gams.md).

Note that the Gurobi license for this interface is separate from the standalone
Gurobi license, and supports far more instances.

!!! tip "Important"
    When using the Gurobi solver in GAMS, the user should NOT load the
    Gurobi module. Simply using "module load gams" will be enough to load the
    required Gurobi components and access rights.
