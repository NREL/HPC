# Advanced Jupyter workshop
Beyond the basics: this advanced Jupyter workshop will survey topics which enable you to get more out of your interactive notebooks. It will build on the recent Intro to Jupyter workshop and introduce additional Magic commands. Interacting with Slurm from a notebook will also be covered, and how this can be used to achieve multi-node parallelism. Additional topics include utilizing GPUs from a notebook, and parameterized notebook execution with Papermill.

Survey format. 45 min slides & demos, 15 min discussion

**Topic 1**: Accessing Slurm and multi-node parallelism

* Magic commands
	* Line vs cell magics
	* Environment query and manipulation, profiling, and language magics
* Slurm commands (Tim)—srunning from a notebook, job status checks, running MPI-enabled routines. 
	* Explain pip install slurm_magic from inside notebook
	* See https://github.com/NREL/HPC/blob/master/languages/python/jupyter/dompi.ipynb
* multi-node parallelism—mpi4py, IPy parallel? (Chris or Tim?)
	* maybe demo timing magics wrt parallel routines

**Topic 2**: Multi-node GPU computing from notebooks

* Accessing GPU compute (Chris)
	* Tim has MPI Tensorflow MNIST Python program. Tim will send link.

**Topic 3**: Papermill
* Parameterized notebook execution
* Overview of steps to run a notebook with Papermill
* Example of predicting MNIST digits 