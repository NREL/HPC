# Demo - Run GAMS Optimization Problems From Python Using GAMS Python API on Eagle

This repo contains python script examples with GAMS optimization problems.

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

## 5. Load Solver

Load the optimization solver you would like to use.

```
module load gurobi/9.0.2
```

# Run Experiments (Solve GAMS Optimization Problems)

The repo contains the transport and indus89 GAMS optimization problems (examples provided by GAMS). Use the following command to solve the trnsport.gms problem (transport.py) using the Gurobi solver via the GAMS Python API.

```
python transport.py
```

After the completion of the solver run, you can find the problem status flags and solution in the log file saved in the same directory (.demos).