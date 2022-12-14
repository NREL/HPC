# Advanced Jupyter
Beyond the basics: this advanced Jupyter directory builds upon our Intro to Jupyter documentation to enable you to get more out of your interactive notebooks. The following topics are covered in this documentation:

**Jupyter Magic commands**

* Magic commands
	* Line vs cell magics
	* Environment query and manipulation, profiling, and language magics

**Accessing Slurm and multi-node parallelism**

* Slurm commands: `srun` from a notebook, job status checks, running MPI-enabled routines. 
	* Explain `pip install slurm_magic` from inside notebook
	* See https://github.com/NREL/HPC/blob/code-examples/general/Jupyterhub/jupyter/dompi.ipynb
	* Demonstration of using slurm magics to run MNIST 
* multi-node parallelism with mpi4py

**Multi-GPU computing from notebooks**
* Accessing GPU compute
	* MPI Tensorflow MNIST Python program which runs on multiple GPUs.

**Papermill**
* Parameterized notebook execution
* Overview of steps to run a notebook with Papermill
* Example of predicting MNIST digits 
