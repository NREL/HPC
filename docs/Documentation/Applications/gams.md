# Using the General Algebraic Modeling System

*The General Algebraic Modeling System (GAMS) is a commercial high-level modeling system for mathematical programming and optimization. It is licensed software.*

GAMS includes a DSL compiler and also a stable of integrated high-performance solvers. GAMS is able to solve complex, large-scale modeling problems. For documentation, forums, and FAQs, see the [GAMS website](https://www.gams.com/).

A 40-user license of GAMS is made available to Eagle users. The GAMS license requires users to be a member of a "workgroup." If you need the GAMS software or a specific solver from GAMS, or if you have trouble running GAMS, please [contact us](https://www.nrel.gov/hpc/contact-us.html).

## Initializing Your Environment

To initialize your environment to use GAMS, simply type `module load gams/<version>` — see `module avail gams` output to see available versions. GAMS is run with the command format `gams <input filename>`. A file `<input filename>.lst` will be created as the output file.   

For a test run, in your home directory, type the following:

```
module load gams/<version>
cp /nopt/nrel/apps/gams/example/trnsport.gms .
gams trnsport
```

A result of 153.675 should be found from screen output. More detailed output is in the file `trnsport.lst`. 

## Selecting an Alternative Solver

The available solvers for different procedures are shown in the following with the default solver being the first one:

* LP: **GUROBI** BDMLP CBC IPOPT SOPLEX
* MIP: **GUROBI** BDMP CBC SCIP
* RMIP: **GUROBI** BDMLP CBC IPOPT SOPLEX
* NLP: **SCIP** COUENNE IPOPT
* MCP: **NLPEC** MILES
* MPEC: **NLPEC**
* RMPEC: **NLPEC**
* CNS: **SCIP** COUENNE IPOPT 
* DNLP: **SCIP** COUENNE IPOPT 
* RMINLP: **SCIP** COUENNE IPOPT 
* MINLP: **SCIP** BONMIN COUENNE 
* QCP: **GUROBI** COUENNE IPOPT SCIP
* MIQCP: **GUROBI** BONMIN COUENNE SCIP
* RMIQCP: **GUROBI** COUENNE IPOPT SCIP
* EMP: **JAMS** LOGMIP SELKIE 

By typing `gams <input_filename>` on the command line, the default procedure LP and the default solver Gurobi will be used. In order to override the default option to use, *e.g.*, Soplex, you can try the following two methods: 

[1]  Use the *option* statement in your GAMS input file. For example, if your model input uses LP procedure and you want to use Gurobi solver to solve it, just add `option lp=soplex` to your input file.

or

[2]  Specify the solver in the command line, *e.g.*, `gams <input_filename> lp=soplex`. 

A sample script for batch submission is provided here:

??? example "Eagle Sample Submission Script"

	```
	#!/bin/bash --login
	#SBATCH --name gams_run
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=36
	#SBATCH --time=00:05:00
	#SBATCH --account=<allocation-id>
	#SBATCH --error=gams-%j.err
	#SBATCH --output=gams-%j.out
	 
	# Ensure script location
	cd $SLURM_SUBMIT_DIR
	 
	# Create runtime environment
	module purge
	module load gams/<version>
	  
	# Run GAMS
	gams trnsport lp=gurobi
	```

For a certain solver, necessary control parameters for the algorithm—such as convergence criteria—can be loaded from the option file named as `<solver_name>.opt` in the directory that you run GAMS. For example, for the Gurobi solver, its option file would be *"gurobi.opt"*. For the details of how to set those parameters, please see the [GAMS Solver Manuals](https://www.gams.com/latest/docs/S_MAIN.html). 

!!! tip "Important"
	 When using the Gurobi solver in GAMS, the user should NOT try to load the Gurobi module. Simply using "module load gams" will automatically load the Gurobi solver.

## Using GAMS Python API

In order to use GAMS python API, the environment parameter `$PYTHONPATH` should include these two directories: 

`$GAMS_PYTHON_API_FILES/gams`
`$GAMS_PYTHON_API_FILES/api_[version-of-python]`

where `version-of-python` =  27, 36, 37, or 38 for python version 2.7, 3.6, 3.7, or 3.8, respectively. The python version can be obtained by using command `python --version`. 

For example, for python 3.7 and the bash shell, `$PYTHONPATH` can be set using the following script:

```
module purge
module load gams/31.1.0
if [ -z ${PYTHONPATH+x} ]
then
        export PYTHONPATH=$GAMS_PYTHON_API_FILES/api_37:$GAMS_PYTHON_API_FILES/gams
else
        export PYTHONPATH=$GAMS_PYTHON_API_FILES/api_37:$GAMS_PYTHON_API_FILES/gams:$PYTHONPATH
fi
```
