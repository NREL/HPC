# Demo - Run GAMS Optimization Problems From Python Using GAMS Python API on Eagle

This repo contains GAMS source code and python script examples to demonstrate the use of the GAMS Python API on Eagle.

# Configuration

## 1. Allocate an interactive Eagle node

First, allocate an interactive node. Let's start by allocating a compute node by using the command below. 

```
srun --time=t0 --account=<project_name> --pty $SHELL
```
where, t0 is the amount of time requisted in minutes and project_name is your project account name on Eagle.

Successful node allocation is shown as:

```
srun: job 7243457 queued and waiting for resources
srun: job 7243457 has been allocated resources
```

## 2. Activate the Anaconda environment:

Activate the Anaconda environment you created (..how_to_guide/gams_python_api_how_to_guide.md)).

```
module purge
conda activate /scratch/$USER/conda-envs/gams_python
```

## 3. Load GAMS and Set up Environment Parameters

```
module purge
module load gams/34.3.0
```

## 4. Set up Environment Parameter

```
export PYTHONPATH=$GAMS_PYTHON_API_FILES/api_37:$GAMS_PYTHON_API_FILES/gams
```

# Run Experiments (Solve GAMS Optimization Problems)

This repo contains the following two examples.
- A Transportation Problem 
  - It finds a least cost shipping schedule that meets requirements at markets and supplies at factories.
  - The GAMS source code (optimization problem code) for this problem is available at .gams_files/transport.gms, relative to the current directory.
  - The python script for running this GAMS problem is /transport.py in the current directory. The GAMS solver is set to be "xpress" for this example.
  - Use the following command, in the current directory, to solve the transport.gms problem using the Xpress solver via the GAMS Python API.
  ```
  python transport.py
  ```
  - After the completion of the solver run, you can find the problem status flags such as objective function value,solve status (solved or not), solver used, etc. in the log file saved with same file name in the same directory  (.demos/transport.log).
  - You can also find more details about the problem and solution found in .lst file that will be automatically created after the solve is completed. .lst file is saved in the GAMS source code directory (.gams_files/_gams_py_gjo0.lst).

- An Optimal Power Flow (OPF) Problem 
  - Multi-period DC-OPF for IEEE 24-bus rts network considering wind and load shedding.
  - The GAMS source code for this problem is available at .gams_files/multi_period_dc_opf.gms.
  - The python script for running this GAMS problem is /multi_period_dc_opf.py in the current directory. The GAMS solver is set to be "gurobi" for this example.
  - Use the following command to solve the multi_period_dc_opf.gms problem using the Gurobi solver via the GAMS Python API.
  ```
  python multi_period_dc_opf.py
  ```
  - You can find the .log and .lst files with the same way as the first example.
