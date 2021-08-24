# Introduction to Pyomo
[Pyomo](http://www.pyomo.org/) is a Python-based and open-source
optimization modeling library, similar in capability to tools such
as [AMPL](https://ampl.com/), [AIMMS](https://www.aimms.com/),
[GAMS](https://www.gams.com/), and [JuMP](https://jump.dev/).
This turorial will focus on aspects of Pyomo specific to its installation
and successful use on Eagle. Pyomo's extensive documentation
is available [here](https://pyomo.readthedocs.io/en/stable/index.html);
users new to optimization modeling should read the [Pyomo introduction
to mathematical modeling](https://pyomo.readthedocs.io/en/stable/pyomo_overview/math_modeling.html).

While Pyomo allows for the expression of optimization models,
their basic solution is off-loaded to optimization solver libraries,
which are typically implemented in lower-level languages such as
C or C++. One important aspect the of successfully utilizing
Pyomo is having access to and properly installing these solver
libraries.

Additionally, Pyomo is the basis for
[several modeling extensions](https://pyomo.readthedocs.io/en/stable/modeling_extensions/index.html),
some of which ship with Pyomo while others are standalone.
Most of these extensions have meta-solvers or transformations
which allow the utilization of more traditional optimization
solvers while modeling in a more abstract fashion. Pyomo is
also the basis of several other software packages, including
[IDAES Process Systems Engineering Framework](https://github.com/IDAES/idaes-pse).
A full list of Pyomo-related packages is available
[here](https://pyomo.readthedocs.io/en/latest/related_packages.html).

# Python
Current versions of Pyomo support Python 3.6+.
Here we use Python 3.8 as our baseline;
[several](https://github.com/Pyomo/mpi-sppy)
[modern](https://github.com/or-fusion/pao)
[pyomo-based](https://github.com/cog-imperial/galini)
[Python](https://github.com/parapint/parapint)
[libraries](https://github.com/grid-parity-exchange/egret)
[require](https://github.com/grid-parity-exchange/Prescient)
Python 3.7+.
```
module load conda
conda create -n virtual_env_name python=3.8
```
which will create an anconda environment using Python 3.8
named `virtual_env_name`. You should use your own name here.
More details for using conda on Eagle can be found [here](../../../general/intermediate/software-environment-basics/conda-how-to.md).

# Installing Pyomo
The lastest version of Pyomo can be installed via `pip`:
```
pip install pyomo
```
Unless there is an explicit backwards capability issue,
users should generally use the latest release of Pyomo.

# Creating a Pyomo Model
This tutorial is *not* about mathematical optimization modeling.
In addition to the [Pyomo documentation](https://pyomo.readthedocs.io/en/stable/),
an excellent resource for mathematical optimization modeling
in Pyomo is [Prof. Jeffrey Kantor's Pyomo Cookbook](https://jckantor.github.io/ND-Pyomo-Cookbook/).
That said, for completeness we give a breif overview of mathematical
optimization modeling.

In general, one can think of an mathematical optimization model as having
five major components:
* Sets
* Parameters
* (decision) Variables
* Constraints
* Objective (function)

Pyomo has modeling objects for each of these components (as well as a few extra).
Below we demonstrate their use on a the
[p-median problem](https://en.wikipedia.org/wiki/Facility_location_problem) adapted from
[this example](https://github.com/Pyomo/PyomoGallery/blob/master/p_median/p-median.py)
utilizing a `ConcreteModel` and demonstrating some of the modeling flexibility in Pyomo.
This example is also available as a [stand-alone python module](./p_median.py).
```python
import pyomo.environ as pyo
import random

random.seed(42)

# These could also be pyo.Param objects
number_of_candidates = 10 # Number of candidate locations
number_of_customers = 6   # Number of customers
number_of_facilities = 3  # Number of facilities 

## Create the model
model = pyo.ConcreteModel()

## Sets
# Set of candidate locations (could be just a pyo.Set if names are better)
model.candidate_locations = pyo.RangeSet(1,number_of_candidates)

# Set of customer nodes
model.customer_nodes = pyo.RangeSet(1,number_of_customers)

## Parameters
# demand[j] - demand of customer j
model.demand = pyo.Param(model.customer_nodes, initialize=lambda m,j : random.uniform(5.0,10.0))

# cost[i,j] - cost of satisfying a unit of demand for customer j from facility i
model.cost = pyo.Param(model.candidate_locations, model.customer_nodes,
        initialize=lambda m,i,j : random.uniform(1.0,2.0))

## (decision) Variables
# fraction_production[i,j] - fraction of demand of customer j that is supplied byfacility i
model.fraction_production = pyo.Var(model.candidate_locations, model.customer_nodes, bounds=(0.,1.))

# build_facility[i] - a binary variable that is 1 if a facility is located at location i
model.build_facility = pyo.Var(model.candidate_locations, within=pyo.Binary)

## Constraints
# Exactly p facilities are located
def facilities_rule(m):
    return sum(m.build_facility[i] for i in m.candidate_locations) == number_of_facilities
model.facilities_limit = pyo.Constraint(rule=facilities_rule)

# All the demand for customer j must be satisfied (with indexed object slice)
def demand_satisfaction_rule(m, j):
    return sum(m.fraction_production[:,j]) == 1.
model.demand_satisfaction = pyo.Constraint(model.customer_nodes, rule=demand_satisfaction_rule)

# Using the @Constraint decorator
# Creates constraint with the same name as the decorated function
@model.Constraint(model.candidate_locations, model.customer_nodes)
def open_demand_served(m, i, j):
    return m.fraction_production[i,j] <= m.build_facility[i]

## Objective 
model.total_cost = pyo.Objective(
        expr=sum(model.demand[j]*model.cost[i,j]*model.fraction_production[i,j]
                for i in model.candidate_locations for j in model.customer_nodes))
```
While the above code *creates* an optimization model in Pyomo, one needs
to use an external solver (linked through Pyomo) to *solve* or *optimize*
this model. 

A more complex example `ConcreteModel` utilizing data brought in from a json
file is available [here](https://github.com/power-grid-lib/pglib-uc/blob/master/uc_model.py).


# Solvers
Pyomo is an Algebraic Modeling Language (AML) implemented
in Python. It does *not* do basic mathematical optimization,
but rather provides interfaces to mathematical optimization 
solvers. The type of solver you need depends on the structure
of your optimization problem.

## (Mixed-Integer) Linear Optimization
Eagle comes configured with two Linear Programming (LP) and
Mixed-Integer (Linear) Programming (MIP) solvers: Xpress and
Gurobi. The functionality of both packages, and their interfaces
with Pyomo, are largely similar. Because of the limited nature of
Gurobi licenses on Eagle, it is recommend to use Xpress unless
utilizing Gurobi-specific features (e.g., callback interfaces with Pyomo).

For both solvers, it is useful to have the Python binding for
that solver installed.
This allows the user (or a Pyomo-based library, such as
[mpi-sppy](https://github.com/Pyomo/mpi-sppy) or
[Prescient](https://github.com/grid-parity-exchange/Prescient))
to exploit the `PersistentSolver` feature of
Pyomo, enabling incremental and rapid model updates and re-solves.
For Xpress, the Python bindings are *required* as the traditional
file-based interface needs maintenance to bring it up-to-date with
current versions of Xpress.

### Xpress
Eagle does system-wide access to Xpress licenses, making it the MILP solver
of choice for use on Eagle. However, the version typically
available is not the latest. One can access the Xpress license by loading
the Xpress module:
```
module load xpressmp
```
For Xpress, it is necessary that you also install the Python
bindings; at the time of writing the LP file interface between Pyomo and
Xpress is largely broken.

The Python bindings for Xpress can be installed via pip:
```
pip install "xpress<8.9"
```
Here we specify the latest version of Xpress (at time of writing) that
is compatible with the license on Eagle, i.e., a version in the 8.8.x series.
You can see the versions of Xpress available on Eagle by running
```
module avail xpressmp
```

#### Verifying the installation
Working with the Pyomo ConcreteModel `model` above, we can verify our
installation of Xpress by calling the `xpress_direct` solver:
```python
xpress = pyo.SolverFactory('xpress_direct')
result = xpress.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with Xpress")
else:
    print("Something went wrong; see message above")
```
An example using Xpress to solve the p-median problem is available
[here](./solve_xpress.py).

### Gurobi
NREL's Gurobi license allows for a limited number of concurrent
Gurobi environments on Eagle. Basic access to Gurobi can be obtained by loading the
Gurobi module
```
module load gurobi
```
The Python bindings for Gurobi can be installed via conda:
```
pip install "gurobipy<10"
```
Here we specify the latest version of Gurobi (at time of writing) that
is compatible with the license on Eagle, i.e., a version in the 9.x series.
You can see the versions of Xpress available on Eagle by running
```
module avail gurobi 
```

#### Verifying the installation
Working with the Pyomo ConcreteModel `model` above, we can verify our
installation of Gurobi by calling the `gurobi_direct` solver:
```python
gurobi = pyo.SolverFactory('gurobi_direct')
result = gurobi.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with Gurobi")
else:
    print("Something went wrong; see message above")
```
An example using Gurobi to solve the p-median problem is available
[here](./solve_gurobi.py).

#### Checking available Gurobi licenses
You can see how many Gurobi licenses are available
using the `gurobi_cl` command:
```
gurobi_cl --tokens
```
expected output:
```
Checking status of Gurobi token server 'license-1.hpc.nrel.gov'...

Token server functioning normally.
Maximum allowed uses: 24, current: 0

Found 1 active servers
```

## Non-Linear Optimization
While Pyomo has access to several non-linear optimization solvers,
none are available (directly) through Eagle modules (several are available
through the GAMS interface detailed next). However, the IPOPT solver
is freely available and is failry robust. IPOPT is a local optimization
solver, meaning for non-convex problems it finds local, as opposed to global,
optima.

IPOPT interfaces with general linear solvers such as MUMPS or the
HSL libraries, and its performance is largely tied to those libraries.

### Installing IPOPT via conda with MUMPS
IPOPT distributed via conda, and can be easily installed as:
```
conda install -c conda-forge ipopt
```
That said, this version is distributed with MUMPS, and is generally
not a performant as version of IPOPT built against an HSL library.

### Installing IPOPT with HSL Mathematical Software Library from IDAES
An IPOPT executable with the MA27, MA57, and MA97 HSL libraries, as well as METIS,
is distributed as part of the IDAES process systems engineering platform.

#### Direct download
An IPOPT executable complied against the HSL library is available under the `idaes-solvers`
archive available [here](https://github.com/IDAES/idaes-ext/releases). Placing
the `ipopt` executable somewhere in your `$PATH` (such as `$HOME/bin`) will allow
pyomo to pick it up.

#### Installing IDAES
Alternatively, IPOPT can be obtained simply by installing IDAES and fetching
the IDAES extensions:
```
pip install idaes-pse
idaes get-extensions
```
Finally you need to add the IDAES solver installation folder to your path:
```
echo export PATH=\"\$PATH:\$HOME/.idaes/bin\" >> ~/.bashrc
```
You should **log out and log back in** for the changes to take effect.

In addition to IPOPT, the IDAES extension library also includes binaries for
several other open-source solvers:
* [CBC](https://github.com/coin-or/Cbc), for mixed-integer linear optimization problems
* [BONMIN](https://github.com/coin-or/Bonmin), for convex mixed-integer
non-linear optimization problems (works as a heuristic for non-convex)
* [COUENNE](https://github.com/coin-or/Couenne), for nonconvex mixed-integer
non-linear optimization problems.

#### Build IPOPT with HSL from source
Follow the directions [here](../../julia/how-to-guides/install-HSL-and-Ipopt.md).

### Verifying IPOPT
At the terminal, try
```
ipopt -v
```
The expected output is something like:
```
Ipopt 3.13.2 (x86_64-pc-linux-gnu), ASL(20190605)
```
Like the other solvers, you can also verify thier installation
in pyomo. Note that IPOPT *does not* solve problems with integer or
binary variables; they get converted to continuous variables when
sent to IPOPT.
```python
ipopt = pyo.SolverFactory('ipopt')
result = ipopt.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median *relaxation* with IPOPT")
else:
    print("Something went wrong; see message above")
```
An example using IPOPT to solve the p-median relaxation is available
[here](./solve_ipopt.py).

## Using GAMS
For all types of optimization problems, Eagle also has
[limited access](https://www.nrel.gov/hpc/eagle-software-gams.html) to the
[General Algebraic Modeling System (GAMS)](https://www.gams.com/),
which contains linked solvers. Note that many of the solvers available
to GAMS are easily available through other means: Gurobi can be
accessed directly through its module; IPOPT, BONMIN, COUENNE, and CBC
are available as part of the IDAES solver library referenced above.

To initialize GAMS, type:
```
module load gams
```
For more on using GAMS on Eagle, see the [Eagle documentation page
for GAMS](https://www.nrel.gov/hpc/eagle-software-gams.html).

### Verifying GAMS
Pyomo contains a generic interface to GAMS available through the GAMS
"solver":
```python
gams = pyo.SolverFactory('gams')
result = gams.solve(model, tee=True)
if pyo.check_optimal_termination(result):
    print("Successfully solved p-median with GAMS")
else:
    print("Something went wrong; see message above")
```
An example using GAMS to solve the p-median problem is available
[here](./solve_gams.py).
