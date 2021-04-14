# Advanced Jupyter workshop

Survey format. 45 min slides & demos, 15 min discussion

Topic/Notebook 1: Accessing Slurm and multi-node parallelism

* demo magics (Kevin or Chris)
	* line vs cell magics
	* environment manipulation/reporting magics
* Slurm commands (Tim)—srunning from a notebook, job status checks, running MPI-enabled routines. 
	* Explain pip install slurm_magic from inside notebook
	* See https://github.com/NREL/HPC/blob/master/languages/python/jupyter/dompi.ipynb
* multi-node parallelism—mpi4py, IPy parallel? (Chris or Tim?)
	* maybe demo timing magics wrt parallel routines

Topic/Notebook 2: Multi-node GPU computing from notebooks

* Accessing GPU compute (Chris)
	* Tim has MPI Tensorflow MNIST Python program. Tim will send link.

Topic 3: Running notebooks without a GUI
* Papermill (Kevin). Maybe notebook 2, searching through hyper parameters with Papermill?
