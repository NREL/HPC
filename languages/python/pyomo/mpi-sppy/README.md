# mpi-sppy

First, you should follow the instructions for installing Pyomo.

# Installing MPICH on Eagle

## TODO: Figure out if this is still best practice, or if OpenMPI4 should be used
mpi-sppy utilizes one-sided memory windows for communication. While
it is not difficult to configure this work on a single node, 
getting multiple nodes working together is a challenge.

As far as I can ascertain, the intel-mpi implementation only
supports shared memory windows on a shared memory system.

OpenMPI does support shared memory windows across nodes, but
I have not been successful finding a set of arguments that
enables this on the versions of OpenMPI available on Eagle.
It's possible it works with OpenMPI >= 4, I have not tested it.

However, what does seem to work in a robust way is MPICH, which
is conveniently available through conda. Conda also has a version
of `mpi4py` complied against it; both can be installed via:
```
conda install mpi4py
```

# Installing mpi-sppy

The easiest way to install mpi-sppy is to clone it:
```
git clone --depth=1 http://github.com/Pyomo/mpi-sppy.git
```
and run setup:
```
cd mpi-sppy; pip install .
```
General documentation and examples for mpi-sppy is available
[here](https://mpi-sppy.readthedocs.io/en/latest/).

While mpi-sppy can also be obtained from PyPI using
`pip install mpi-sppy`, installing from source give easy access
to both the included examples, as well as the verification
script described next.


# Verifying your installation
Because HPC computing environments vary, mpi-sppy ships with a test
utility to verify the one-sided windows are behaving as intended; i.e.,
that their use is non-blocking.

If you have installed MPICH as directed above, it is important to
always have the follow environment variable set:
```
export MPICH_ASYNC_PROGRESS=1
```
This enables MPICH to communicate the memory windows without blocking
between nodes. 

As I understand it, MPICH starts "watcher threads" to
transport the shared-memory buffers across nodes. On the problems
I've tested, there isn't a noticeable performance impact for doing so.

Below is an example slurm script for verifying your installation,
which is also available in `./slurm/`:
```
#!/bin/bash
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks=2                # Request 2 CPU cores
#SBATCH --time=00:05:00             # Job should run for up to 5 minutes
#SBATCH --account=aces  	# Where to charge NREL Hours
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=Bernard.Knueven@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=mpisppy_test.%j.out  # %j will be replaced with the job ID

source ~/.bashrc
#module load intel-mpi
conda activate mpisppy-conda-mpich

#export OMPI_MCA_btl=openib,self
export MPICH_ASYNC_PROGRESS=1

cd ${HOME}/software/mpi-sppy
srun python -m mpi4py mpi_one_sided_test.py
```
Note that this script asks for 2 nodes on Eagle, to verify that cross-node
one-sided communication is tested as intended. If this script produces no
output, that is generally a good sign that your configuration does not have
blocking one-sided windows. If the one-sided windows *are* blocking, then
`mpi_one_sided_test.py` will raise an assertion error.


# Example slurm scripts
Example slurm scripts using the unit commitment problem are available in 
`./slurm/uc/`. These use a working directory `mpi-sppy/paperruns/larger_uc/`
as a base. Below we show the 1000-scenario UC slurm script (1000scen\_fw):
```
#!/bin/bash
#SBATCH --nodes=223                   # Number of nodes
#SBATCH --ntasks=4000                # Request 4000 CPU cores
#SBATCH --time=00:10:00             # Job should run for up to 10 minute
#SBATCH --account=aces  	# Where to charge NREL Hours
#SBATCH --mail-user=Bernard.Knueven@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=1000scen_fw.%j.out  # %j will be replaced with the job ID

source ~/.bashrc
#module load intel-mpi
conda activate mpisppy-conda-mpich

#export OMPI_MCA_btl=openib,self
export MPICH_ASYNC_PROGRESS=1

cd ${HOME}/software/mpi-sppy/paperruns/larger_uc

srun python3.7 uc_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1.0 --num-scens=1000 --max-solver-threads=2 --solver-name=xpress_persistent --rel-gap=0.00001 --abs-gap=1 --no-cross-scenario-cuts --with-display-timing --intra-hub-conv-thresh=-1.0
```

When running several instances of an optimization solver (in this case, Xpress)
in parallel, it is important to configure the solver to only use a limited number
of threads so as not to overload the individual nodes. For example, here we request
223 nodes, or (223 nodes X 36 threads/node = ) 8028 threads, but only 4000 tasks.
However, we specify that the solver Xpress can use up to 2 threads `--max-solver-threads=2`,
which when all the solvers are running nearly maxes out our allocation request with 
(4000 task X 2 threads/task = ) 8000 threads.

WARNING: By default most optimization solvers will use all the available threads on a node
(or a large fraction of them). Therefore it is very important for performance to limit the
threads an optimization solver can use such that you are not over-subscribing your allocated
resources.

# Using unbuffer 
It might be necssary to use unbuffer to debug certain MPI issues.
Due to existing issues when using `unbuffer` with conda environments
on Eagle, the most straightfoward thing to do is launch python in
unbuffered mode, e.g.:
```
srun python -u -m mpi4py mpi_one_sided_test.py 
```
which will flush mpi-sppy's logs to stdout and stderr immediately.
(Useful for approximately knowing when a thread may be hanging.)
