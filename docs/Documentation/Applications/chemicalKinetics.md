---
title: Chemical-Kinetics
parent: Applications
---

# Chemical kinetics: Cantera, zero-RK, PelePhysics
<!---
**Documentation:** [ link to documentation](https://nrel.gov)
-->
*Chemical kinetics packages are tools for problems involving chemical kinetics, thermodynamics, and transport processes. They can be useful in applications including but not limited to combustion, detonations, electrochemical energy conversion and storage, fuel cells, batteries, aqueous electrolyte solutions, plasmas, and thin film deposition.*

## Overview

A wide variety of packages are available for the purpose, each with their strengths, pros and cons. The matrix below provides a birds eye view of some of the packages. When applicable, please refer to the footnotes marked at the bottom of the page. (All company, product and service names used on this page are for identification purposes only. Use of these names, trademarks and brands does not imply endorsement.)

|                                                                          | C++   | Fortran | Python | Matlab | GPU    | Speed*$^5$ | Features | Cost | Compatibility       | Speciality                   | Notes      |
|:------------------------------------------------------------------------:|:-----:|:-------:|:------:|:------:|:------:|:----------:|:--------:|:----:|:-------------------:|:----------------------------:|:----------:|
| [Cantera](https://cantera.org/)                                          | y     | y       | y      | y      | x      | ++         | ++++     | Free | Research codes*$^8$ | Simplicity, large user base  | 0.0.0      |
| [zero-RK](https://github.com/LLNL/zero-rk)                               | y     | x       | x*$^6$ | x      | y*$^1$ | ++++*$^7$	 | ++*$^4$  | Free | Converge CFD ($)    | Model reduction tools        | 0.0.0 	 |  
| [PelePhysics](https://amrex-combustion.github.io/PelePhysics/)           | y     | x       | x      | x      | y      | +++++      | +++      | Free | Amrex/Pele          | HPC, NREL popular framework  | 0.0.0      |
| [Chemkin Pro](https://www.ansys.com/products/fluids/ansys-chemkin-pro)   | y     | y*$^2$  | x      | x      | x*$^3$ | ++++       | ++++     | $    | Ansys ($)           | Legacy, professional support | 0.0.0      |


## Strategy
A typical workflow could look as follows:

1. Create mechanisms and validate with Cantera:

	👍 Feature rich, multi language, very [well documented](https://cantera.org/documentation/index.html), large [support forum](https://groups.google.com/g/cantera-users).

	👎 Slower than competition, no GPU support.


2. Perform model reduction with zero-RK if necessary:

	👍  Faster than Cantera & Chemkin for [more than 100 species](https://ipo.llnl.gov/sites/default/files/2019-09/zork.pdf), some [GPU support](https://doi.org/10.1115/ICEF2017-3631).

	👎 Fewer features, sparse documentation, C++ only.

3. Convert to a high performance C++ code with PelePhysics and link to a CFD solver: 

	👍  [GPU](https://amrex-combustion.github.io/PelePhysics/CvodeInPP.html#cvode-implementation-in-pelephysics-on-gpu), [well documented](https://amrex-combustion.github.io/PelePhysics/index.html), [amrex/Pele CFD](https://amrex-combustion.github.io/) & Cantera mechanisms compatible.
	
	👎 Fewer features than Cantera & Chemkin, C++ only. 

## Installation and testing on Kestrel

### Cantera
!!! note
	Cantera can be [installed from source](https://cantera.org/install/compiling-install.html#sec-compiling) or with the conda envionment on the Kestrel as explained below.
	The performance can vary depending on the compile options and flags while compiling from source. We are more than happy to learn from power users about the flags which lead to the best performance. 
[Installation: Python version](https://cantera.org/install/conda-install.html#sec-install-conda)
```
$ module load conda
$ conda create --name ct-env --channel cantera cantera ipython matplotlib jupyter
$ conda activate ct-env

$ python3
>>> import cantera as ct
>>> ct.one_atm
101325.0
>>> exit() 
$ conda deactivate
```
??? example "[Example interactive jupyter usage](https://nrel.github.io/HPC/Documentation/Development/Jupyter/)"
	```
	$ ssh username@kestrel.hpc.nrel.gov
	```
	To access your scratch from the jupyter notebook, execute the following from your Kestrel home directory (optional)
	```
	$ ln -s /scratch/username scratch
	```
	```
	$ module load conda
	$ conda activate ct-env
	```
	Create a jupyter kernel from ct-env
	```
	$ python -m ipykernel install --user --name=ct-env
	```
	In a browser, go to [Kestrel JupyterHub](https://kestrel-jhub.hpc.nrel.gov/), select “ct-env” in the Notebook section to open a new jupyter notebook with the ‘ct-env’ loaded

	Try Cantera python API within the notebook, for example,
	```
	import cantera as ct
	ct.one_atm
	```

[Installation: C++ version](https://cantera.org/install/conda-install.html#sec-conda-development-interface)
```
$ module load conda
$ conda create --name ct-dev --channel cantera libcantera-devel
$ conda activate ct-dev
$ conda install cmake scons pkg-config

$ cd /home/username/.conda-envs/ct-dev/share/cantera/samples/cxx/demo
$ scons && ./demo
$ cmake . && cmake --build . && ./demo
$ g++ demo.cpp -o demo $(pkg-config --cflags --libs cantera) && ./demo
```

??? example "[Example interactive C++ usage](https://www.nrel.gov/hpc/running-jobs.html)"
	```
	$ ssh username@kestrel.hpc.nrel.gov
	```
	Allocate resources
	```
	salloc --account=allocationName --time=00:30:00 --nodes=1 --ntasks-per-core=1 --ntasks-per-node=104 --cpus-per-task=1 --partition=debug
	```
	Load Cantera
	```
	module load conda
	conda activate ct-dev
	```
	Compile your code
	```
	CC -DEIGEN_USE_LAPACKE -DEIGEN_USE_BLAS -fopenmp -O3 -march=native -std=c++17 -I /home/username/.conda-envs/ct-dev/include/cantera/ext -I . mainYourCode.C $(pkg-config --cflags --libs cantera) -o flame.out
	```
	Execute
	``` 
	srun -n 5 ./flame.out
	```
	Please refer to the [job submission documentation](https://www.nrel.gov/hpc/running-jobs.html) for larger jobs in Batch mode.   

### zero-RK
[Please follow the official installation instructions](https://github.com/LLNL/zero-rk).

### PelePhysics
[Please follow the official installation instructions](https://amrex-combustion.github.io/PelePhysics/GettingStarted.html#building-and-running-test-cases).

!!! note
	Please mind the amrex dependency and remember to set the `AMREX_HOME` environment variable to your amrex location before beginning to compile PelePhysics.

<!---
This section provides the minimum amount of information necessary to successfully run a basic job on an NREL Cluster.
This information should be as complete and self-contained as possible.

Instructions should be step-by-step and include copy-and-pastable commands where applicable.

For example, describe how the user can load the program module  with `module avail` and `module load`:

```
module avail program
   program/2.0.0    program/1.0.0
```

```
module load program/2.0.0
```


Include a section on how to run the job, e.g., with job script examples or commands for an interactive session.

### Example Job Scripts

??? example "Kestrel CPU"

	```slurm
	#!/bin/bash

	# In a comment summarize the hardware requested, e.g. number of nodes, 
        # number of tasks per node, and number of threads per task

	#SBATCH --time=
	#SBATCH --nodes=
	#SBATCH --ntasks-per-node=
	#SBATCH --cpus-per-task=
	#SBATCH --partition=
	#SBATCH --account=

	# include a section of relevant export and module load commands, e.g.:

	module load gcc/8.4.0

	export OMP_NUM_THREADS=

	# include a sample srun command or similar
	srun program.x

	```

??? example "Vermillion"

	If the submit script for Vermillion differs from Kestrel, then include a Vermillion example script here.
	If the submit script does not differ, then remove this section (starting from the `??? example "Vermillion"` line)


??? example "Swift"

	If the submit script for Swift differs from Kestrel, then include a Swift  example script here.
	If the submit script does not differ, then remove this section (starting from the `??? example "Swift"` line)


??? example "Template"
	
	Here's a template of a collapsible example.

	```
	You can include blocked sections
	```

	And unblocked sections.

!!! note
	You can use a note to draw attention to information.

Include instructions on how to submit the job script

## Supported Versions

| Kestrel | Swift | Vermillion |
|:-------:|:-----:|:----------:|
| 0.0.0   | 0.0.0 | 0.0.0      |

## Advanced

Include advanced user information about the code here (see BerkeleyGW page for some examples)

One common "advanced case" might be that users want to build their own version of the code.

### Building From Source

Here, give detailed and step-by-step instructions on how to build the code, if this step is necessary. Include detailed instructions for how to do it on each applicable HPC system. Be explicit in your instructions. Ideally a user reading one of the build sections can follow along step-by-step
and have a functioning build by the end.

If building from source is not something anyone would reasonably want to do, remove this section.

Be sure to include where the user can download the source code

??? example "Building on Kestrel"

	Include here, for example, a Kestrel-specific makefile (see berkeleygw example page). This template assumes that we build the code with only one toolchain, which may not be the case. If someone might reasonably want to build with multiple toolchains, use the "Multiple toolchain instructions on Kestrel" template instead.
	
	```
	Include relevant commands in blocks.
	```
	or as in-line `blocks`

	Be sure to state how to set-up the necessary environment, e.g.:

	```
	module load gcc/8.4.0
	module load openmpi/3.1.6/gcc-8.4.0
	module load hdf5/1.10.6/gcc-ompi
	```

	Give instructions on compile commands. E.g., to view the available make targets, type `make`. To compile all program executables, type:

	```
	make cleanall
	make all
	```
	
??? example "Building on Vermillion"

	information on how to build on Vermillion

??? example "Building on Swift"

	information on how to build on Swift


## Troubleshooting

Include known problems and workarounds here, if applicable
-->

## Footnotes

* 1 Not clear from the documentation but ‘gpu’ exists in the code in several places. No actual GPU users amongst those surveyed at the NREL.

* 2 Also possible through Chemkin II, which was a free Fortran library, not available online anymore.

* 3 The Ansys Fluent CFD solver uses GPU, the Chemkin Pro module does not.

* 4 Features unclear due to very poor documentation. Estimate based on reading parts of the code and NREL user comments.

* 5 Very vague estimate from documentation and NREL user comments. Benchmarking not performed.

* 6 Python scripts exist which gather parameters to execute C++ executables, no actual Python / Cython API like Cantera.

* 7 Faster than Chemkin and Cantera for mechanisms involving more than 100 species, information from documentation.

* 8 Coupling with various codes such as OpenFoam, Nek5000, JAX-Fluids etc. has been possible.
