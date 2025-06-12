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

|                                                                          | C++   | Fortran | Python | Matlab | GPU    | Speed*[^5] | Features | Cost | Compatibility       | Speciality                   |
|:------------------------------------------------------------------------:|:-----:|:-------:|:------:|:------:|:------:|:----------:|:--------:|:----:|:-------------------:|:----------------------------:|
| [Cantera](https://cantera.org/)                                          | y     | y       | y      | y      | x      | ++         | ++++     | Free | Research codes*[^8] | Simplicity, large user base  |
| [zero-RK](https://github.com/LLNL/zero-rk)                               | y     | x       | x*[^6] | x      | y*[^1] | ++++*[^7]  | ++*[^4]  | Free | Converge CFD ($)    | Model reduction tools        |  
| [PelePhysics](https://amrex-combustion.github.io/PelePhysics/)           | y     | x       | x      | x      | y      | +++++      | +++      | Free | Amrex/Pele          | HPC, NREL popular framework  |
| [Chemkin Pro](https://www.ansys.com/products/fluids/ansys-chemkin-pro)   | y     | y*[^2]  | x      | x      | x*[^3] | ++++       | ++++     | $    | Ansys ($)           | Legacy, professional support |


## Strategy
A typical workflow could look as follows:

1. Create mechanisms and validate with Cantera:

	🔼 Feature rich, multi language, very [well documented](https://cantera.org/documentation/index.html), large [support forum](https://groups.google.com/g/cantera-users).

	🔽 Slower than competition, no GPU support.


2. Perform model reduction with zero-RK if necessary:

	🔼  Faster than Cantera & Chemkin for [more than 100 species](https://ipo.llnl.gov/sites/default/files/2019-09/zork.pdf), some [GPU support](https://doi.org/10.1115/ICEF2017-3631).

	🔽 Fewer features, sparse documentation, C++ only.

3. Convert to a high performance C++ code with PelePhysics and link to a CFD solver: 

	🔼  [GPU](https://amrex-combustion.github.io/PelePhysics/CvodeInPP.html#cvode-implementation-in-pelephysics-on-gpu), [well documented](https://amrex-combustion.github.io/PelePhysics/index.html), [amrex/Pele CFD](https://amrex-combustion.github.io/) & Cantera mechanisms compatible.
	
	🔽 Fewer features than Cantera & Chemkin, C++ only. 

## Installation and testing on Kestrel
!!! note
	Cantera can also be [installed from source](https://cantera.org/install/compiling-install.html#sec-compiling) apart from the conda method explained below.
	The performance of packages mentioned on this page can vary depending on the choice of dependency library variants and optimization flags while compiling from source. We are more than happy to learn from power users about choices which lead to the best performance. Please report your experiences by [email](mailto:hpc-help@nrel.gov).

!!! Warning
	Conda environments should be *always* be installed outside of your home directory for storage and performance reasons. This is especially important if linking a chemical kinetics package with a C++ code whose parallel processes can strain the `/home` filesystem. Please refer to our dedicated [conda documentation](../../../../Environment/Customization/conda.md#creating-environments-by-location) for more information on how to setup your conda environments to redirect the installation outside of `/home` by default.

### Cantera
 [Installation: Python version](https://cantera.org/install/conda-install.html#sec-install-conda)
```
$ module load anaconda3/2024.06.1
$ conda create --prefix ./ct-env --channel cantera cantera ipython matplotlib jupyter
$ conda activate ./ct-env

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
	$ module load anaconda3/2024.06.1
	$ conda activate ./ct-env
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
$ module load anaconda3/2024.06.1
$ conda create --prefix ./ct-dev --channel cantera libcantera-devel
$ conda activate ./ct-dev
$ conda install cmake scons pkg-config

$ cd /projects/<projectname>/<username>/ct-dev/share/cantera/samples/cxx/demo
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
	module load anaconda3/2024.06.1
	conda activate ./ct-dev
	```
	Compile your code
	```
	CC -DEIGEN_USE_LAPACKE -DEIGEN_USE_BLAS -fopenmp -O3 -march=native -std=c++17 -I /projects/<projectname>/<username>/ct-dev/include/cantera/ext -I . mainYourCode.C $(pkg-config --cflags --libs cantera) -o flame.out
	```
	Execute
	``` 
	srun -n 5 ./flame.out
	```
	Please refer to the [job submission documentation](https://nrel.github.io/HPC/Documentation/Slurm/batch_jobs/) for larger jobs in Batch mode.   

### zero-RK
Please follow the [official installation instructions](https://github.com/LLNL/zero-rk). The procedure has been tested successfully on the Kestrel and repeated below from the official instructions for convenience.
```
$ git clone https://github.com/llnl/zero-rk   #git
$ cd zero-rk
$ mkdir build
$ cd build
$ cmake ../                                   #configure
$ make                                        #build
$ ctest                                       #test
$ make install                                #install
```

### PelePhysics
!!! note
	Please mind the amrex dependency and remember to set the `AMREX_HOME` environment variable to your amrex location before beginning to compile PelePhysics.
Please follow the [official instructions](https://amrex-combustion.github.io/PelePhysics/GettingStarted.html#building-and-running-test-cases) for obtaining the PelePhysics library and compiling examples. The procedure has been tested successfully on the Kestrel. The process for obtaining PelePhysics and compiling the ReactEval example is repeated below from the official instructions for convenience.
```
$ git clone --recursive https://github.com/AMReX-Combustion/PelePhysics.git
$ cd PelePhysics
$ git pull && git submodule update
$ cd Testing/Exec/ReactEval
$ make TPL
$ make
``` 

## Footnotes

[^1]: Not clear from the documentation but ‘gpu’ exists in the code in several places. No actual GPU users amongst those surveyed at the NREL.

[^2]: Also possible through Chemkin II, which was a free Fortran library, not available online anymore.

[^3]: The Ansys Fluent CFD solver uses GPU, the Chemkin Pro module does not.

[^4]: Features unclear due to very poor documentation. Estimate based on reading parts of the code and NREL user comments.

[^5]: Very vague estimate from documentation and NREL user comments. Benchmarking not performed.

[^6]: Python scripts exist which gather parameters to execute C++ executables, no actual Python / Cython API like Cantera.

[^7]: Faster than Chemkin and Cantera for mechanisms involving more than 100 species, information from documentation.

[^8]: Coupling with various codes such as OpenFoam, Nek5000, JAX-Fluids etc. has been possible.
