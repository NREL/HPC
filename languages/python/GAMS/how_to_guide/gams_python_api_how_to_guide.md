# Create Anaconda Environment

First, create an Anaconda environment that you will use for all your experiments related to gams in python. Follow the steps below.

### 1. Log in to Eagle

```
ssh <username>@eagle.hpc.nrel.gov
```
where, 'username' is your NREL HPC (Eagle) user name.

### 2. Set up Anaconda environment

The repo provides the gams_python_api.yml file. Use it to create a new Anaconda environment at a directory of your choose. 

There are three main directories on Eagle where you can install the new environment, namely /home, /scratch, and /projects. Please go to NREL HPC resources page to find more information about the various Eagle directories and how to create new Anaconda environments.

Example:

Start by creating a subdirectory /scratch/$USER/github-repos/, cd there and clone this repo. 

Assuming you want to install your new environment in your scratch directory, you may want to create a directory that will contain all your Anaconda environments, e.g. /scratch/$USER/conda-envs/:

```
conda env create --prefix=/scratch/$USER/conda-envs/gams_python -f gams_python_api.yml
```
The installation may take a couple of minutes, wait until it is complete and you get a done message.

# Load GAMS and Set up Environment Parameters

Now, activate your Anaconda environment, load the appropriate GAMS version available on Eagle and establish the required environment varables. Follow the steps below.

### 1. Activate your environment

```
module purge
conda activate /scratch/$USER/conda-envs/gams_python
```

### 2. Load GAMS

```
module purge
module load gams/34.3.0
```

### 3. Set up environment paramater

In order to use GAMS python API, the environment parameter $PYTHONPATH should include these two directories:

- $GAMS_PYTHON_API_FILES/gams
- $GAMS_PYTHON_API_FILES/api_[version-of-python]

where version-of-python = 27, 36, 37, or 38 for python version 2.7, 3.6, 3.7, or 3.8, respectively. The python version can be obtained by using the command:

```
python --version
```

For example, if your python version is 3.7 the environment parameter $PYTHONPATH can be set using the following script: 

```
export PYTHONPATH=$GAMS_PYTHON_API_FILES/api_37:$GAMS_PYTHON_API_FILES/gams
```
### 4. Make sure if you correctly load and set up the GAMS Python API

Open a python session and import the gams package.

```
python
>>> import gams
```
You have correctly loaded and set up the Python GAMS API on Eagle if you don't get any import error message for the above command.

# Load Solvers

Once you load and set up the GAMS Python API, you should then load your choice of solver available on Eagle in order to solve your optimization problem. For example, the Gurobi solver can be loaded by using the command:

```
module load gurobi/9.0.2
```