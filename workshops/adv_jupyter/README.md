# Advanced Jupyter workshop
Beyond the basics: this advanced Jupyter workshop will survey topics which enable you to get more out of your interactive notebooks. It will build on the recent Intro to Jupyter workshop and introduce additional Magic commands. Interacting with Slurm from a notebook will also be covered, and how this can be used to achieve multi-node parallelism. Additional topics include utilizing GPUs from a notebook, and parameterized notebook execution with Papermill.

Survey format. 45 min slides & demos, 15 min discussion

**Topic 1**: Jupyter Magic commands

* Magic commands
	* Line vs cell magics
	* Environment query and manipulation, profiling, and language magics

**Topic 2**: Accessing Slurm and multi-node parallelism

* Slurm commands: `srun` from a notebook, job status checks, running MPI-enabled routines. 
	* Explain `pip install slurm_magic` from inside notebook
	* See https://github.com/NREL/HPC/blob/master/languages/python/jupyter/dompi.ipynb
	* Demonstration of using slurm magics to run MNIST 
* multi-node parallelism with mpi4py

**Topic 3**: Multi-GPU computing from notebooks
* Accessing GPU compute
	* MPI Tensorflow MNIST Python program which runs on multiple GPUs.

**Topic 4**: Papermill
* Parameterized notebook execution
* Overview of steps to run a notebook with Papermill
* Example of predicting MNIST digits 