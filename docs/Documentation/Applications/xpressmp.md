# Xpress Solver

*FICO Xpress Optimizer provides optimization algorithms and technologies to solve linear, mixed integer and non-linear problems*

For documentation, forums, and FAQs, see the [FICO
website](https://www.fico.com/fico-xpress-optimization/docs/latest/overview.html).

The Xpress solver includes algorithms that can solve

* Linear Programs
* Mixed Integer Programs
* Quadratic Programs
* Quadratically Constrained Quadratic Programs
* Second Order Cone Problems

Xpress solver cannot be used to solve nonlinear programs. 


## Available Modules

| Kestrel (CPU)   |
|:---------------:|
| xpressmp/9.0.2 |
| xpressmp/9.2.2 |
| xpressmp/9.2.5 |

!!! info
    Xpress is available as a module on Kestrel. Additionally, NREL has a site-wide license for Xpress to run locally on an NREL-issued computer. Please see instructions [here](https://github.nrel.gov/MSOC/fico-xpress)

## Running Xpress Solver on Kestrel

!!! important
    While Xpress Solver is available as a module on Kestrel for use by all NREL-users, you MUST be a part of `xpressmp` group on Kestrel. If you are new or have not used Xpress in a while, you can:

    1. Check whether you are a part of this group by running the `groups` command from your terminal, or
    2. Load the `xpressmp` module and run an example

    If you are not a part of the `xpressmp` linux group and/or are unable to run an Xpress instance, please submit a ticket to [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov) requesting access to Xpress on HPC systems and provide a business justification that describes how you intend to use Xpress in your workflow.

Xpress solvers can be used by simply loading the module

```bash
module load xpressmp/9.2.5
```

Once the module is loaded, Xpress Solver can be used directly using command line 
by running the `optimizer` command.

```bash
$ optimizer
FICO Xpress Solver 64bit v9.2.5 Nov  9 2023
(c) Copyright Fair Isaac Corporation 1983-2023. All rights reserved
 Optimizer v42.01.04    [/nopt/nrel/apps/software/xpressmp/9.2.5/lib/libxprs.so.42.01.04]
[xpress kpanda] 
```

Alternatively, Xpress can now be used directly in Python or Julia by loading the necessary modules and programming environments.
