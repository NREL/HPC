## How to use conda on Eagle
Anaconda provides python functionality on Eagle. The default environment contains commonly used mathematic and scientific libraries optimized with MKL. Any additional libraries can be installed with a custom environment.

```
module purge; module load conda
```
Table of Contents
1. [Creating a custom environment](#creating-a-custom-environment)
2. [Updating a custom environment](#updating-a-custom-environment)
3. [Using R within conda](#using-r-within-conda)


### Creating a custom environment
Custom environments can be created with [conda create](https://docs.conda.io/projects/conda/en/latest/commands/create.html) or [conda env create](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).  `conda create` accepts package names in the command path, whereas `conda env create` requires the use of an environment.yml file.  This [environment.yml](https://github.nrel.gov/hsorense/conda-peregrine/blob/code-examples/environment.yml) is used to create Eagle's default conda environment.  It can be copied and modified for a custom enviornment.  Be sure to change the name to something other than default or root, or omit it altogether and use the command line option.

The default location for custom environments is $HOME/.conda-envs . A custom directory can be used with the command line options for path and name.  Environments tend to use large amounts of disk.  If you are getting messages about going over the quota but can't find where the usage is, check the environments directory and remove unused ones.  

#### Example environment using conda create
In this example, a new conda environment containing only python 3 is created.  Once it is verified to be working, another package is added to the environment and tested.  `$` represents the user's shell prompt, `<snip>` represents removed unimportant output, everything else is output from the command.

```
$ conda create -n py3 python=3
Solving environment: done
<snip>
## Package Plan ##

  environment location: $HOME/.conda-envs/py3

  added / updated specs: 
    - python=3


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    xz-5.2.3                   |       h5e939de_4         365 KB
    sqlite-3.23.1              |       he433501_0         1.5 MB
    python-3.6.5               |       hc3d631a_1        29.4 MB
    certifi-2018.4.16          |           py36_0         142 KB
    ------------------------------------------------------------
                                           Total:        31.4 MB

The following NEW packages will be INSTALLED:

    ca-certificates: 2018.03.07-0     
<snip>
    zlib:            1.2.11-ha838bed_2

Proceed ([y]/n)? y


Downloading and Extracting Packages
<snip>
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use:
# > source activate py3
#
# To deactivate an active environment, use:
# > source deactivate
#

$ source activate py3
($HOME/.conda-envs/py3) $ python --version
Python 3.6.5 :: Anaconda, Inc.
($HOME/.conda-envs/py3) $ conda install pandas
Solving environment: done
<snip>
## Package Plan ##

  environment location: $HOME/.conda-envs/py3

  added / updated specs: 
    - pandas


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    mkl_random-1.0.1           |   py36h629b387_0         373 KB
<snip>
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

($HOME/.conda-envs/py3) $ python -c "import pandas; print(pandas.__version__)"
0.22.0
($HOME/.conda-envs/py3) $ source deactivate
$ python --version
Python 2.7.14 :: Anaconda custom (64-bit)
$ python -c "import pandas; print(pandas.__version__)"
0.21.1
```

#### Example environment using conda env create

In this example, a new `environment.yml` file is created that defines a name, version of python and pandas.  Once the environment is created, it will match the previous example.

```
$ echo "name: py3-env
dependencies:
 - python=3
 - pandas
" > environment.yml
$ conda env create 
Solving environment: done
Downloading and Extracting Packages
<snip>
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate py3-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

$ . activate py3-env
($HOME/.conda-envs/py3-env) $ python --version
Python 3.6.5 :: Anaconda, Inc.
($HOME/.conda-envs/py3-env) $ python -c "import pandas; print(pandas.__version__)"
0.22.0
($HOME/.conda-envs/py3-env) $ source deactivate
$ python --version
Python 2.7.14 :: Anaconda custom (64-bit)
$ python -c "import pandas; print(pandas.__version__)"
0.21.1
```

#### Example environment using high performance openmpi
In this example, a new conda environment containing numpy, mpi4py and high performance openmpi built with infiniband support.  The first create is done from a single conda create command, while the second utilizes an environment.yml file along with conda env create. `$` represents the user's shell prompt, `<snip>` represents removed unimportant output, everything else is output from the command.

```
$ conda create -p /scratch/$USER/conda/myhpompi -c local python=2 mpi4py numpy
Solving environment: done

## Package Plan ##

  environment location: /scratch/$USER/conda/myhpompi

  added / updated specs: 
    - mpi4py
    - numpy
    - python=2


The following NEW packages will be INSTALLED:

    binutils_impl_linux-64: 2.28.1-had2808c_3            
    <snip> 
    mkl_random:             1.0.1-py27h629b387_0         
    mpi4py:                 3.0.0-py27h14c3975_0    local
    ncurses:                6.1-hf484d3e_0               
    numpy:                  1.14.5-py27hcd700cb_0        
    numpy-base:             1.14.5-py27hdbf6ddf_0        
    openmpi:                2.1.2-h9ac9557_0        local
    openssl:                1.0.2o-h20670df_0            
    <snip>
    zlib:                   1.2.11-ha838bed_2            

Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /scratch/$USER/conda/myhpompi
#
# To deactivate an active environment, use
#
#     $ conda deactivate



```

__environment.yml__
```
channels:
  - local
  - defaults
dependencies:
  - numpy
  - openmpi
  - mpi4py
```

Creating from environment.yml:
```
$ conda env create -p /scratch/$USER/conda/myhpompi
Solving environment: done

Downloading and Extracting Packages
setuptools-39.2.0    |  583 KB | ###################### | 100% 
pip-10.0.1           |  1.7 MB | ###################### | 100% 
python-2.7.15        | 12.1 MB | ###################### | 100% 
wheel-0.31.1         |   62 KB | ###################### | 100% 
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate /scratch/$USER/conda/myhpompi
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### Updating a custom environment

Modules can be added to a conda environment after it has been created. 

#### Updating an environment via `environment.yml`

If you created an environment by calling `conda env create`, you can list additional dependencies in `environment.yml` and call `conda env update` to install the new packages to the environment.

```
$ cat environment.yml
name: example
dependencies:
 - python=3
 
$ echo " - numpy" >> environment.yml
$ conda env update
Solving environment: done

Downloading and Extracting Packages
numpy-base-1.14.5    |  4.1 MB | #################################################################### | 100%
blas-1.0             |   48 KB | #################################################################### | 100%
libopenblas-0.2.20   |  8.8 MB | #################################################################### | 100%
libgfortran-ng-7.2.0 |  1.2 MB | #################################################################### | 100%
numpy-1.14.5         |   94 KB | #################################################################### | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate example-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

$ conda activate example
(example) [user@host ~]$ python
Python 3.7.0 (default, Jun 28 2018, 13:15:42)
[GCC 7.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy
>>>
```

Similarly, modules can be removed from `environment.yml` prior to calling `conda env update` to remove packages from the environment (though you may consider using [`conda clean`](https://docs.conda.io/projects/conda/en/latest/commands/clean.html) for this).

#### Updating an environment via `conda install`

It is also possible to add modules to an environment by calling `conda install <package>`.

```
(example) [user@host ~]$ conda install matplotlib
Solving environment: done

...
```

### Using R within conda

Conda provides many R packages.  The core R library is installed using the package r-essentials.  A list of all available R packages can be obtained with `conda search r-*`.  Additional R packages can be installed from CRAN with the install.packages() function.  Installing pacman is a possibility but some R packages installed with conda may conflict with those installed with pacman.  Add the r-git2r package to the custom environment prior to installing pacman from CRAN.
