---
layout: default
title: Spack 
parent: Python
grand_parent: Programming Languages
---

![conda logo](images/conda_logo.png)


## Why Conda?

Conda is a **package manager** which allows you to easily create and switch betwen different software environments in different languages for different purposes.  With Conda, it's easy to:

* Manage different (potentially conflicting) versions of the same software without complication

* Quickly stand up even complicated dependencies for stacks of software

* Share your specific programming environment with others for reproducible results 


## First things first


In order to use spack, we can clone the spack repositery using 
'git clone -c feature.manyFiles=true https://github.com/spack/spack.git'

It should be noted that the Spack folder needs to be cloned to a folder that does not get purged/cleaned, e.g. /scratch or /home.

The pre-requisits for Spack are listed in the table below and should load to the path using **module load** before using spack.

| Name                | requirement reason                              |
|---------------------|-------------------------------------------------|
|Python               |Interpreter for Spack                            |            
|C/C++ Compilers      |Building software                                | 
|make                 |Build software                                   | 
|patch                |Build software                                   | 
|bash                 |Compiler wrappers                                | 
|tar                  |Extract/create archives                          | 
|gzip                 |Compress/Decompress archives                     |     
|unzip                |Compress/Decompress archives                     |
|bzip2                |Compress/Decompress archives                     |
|xz                   |Compress/Decompress archives                     | 
|zstd                 |Compress/Decompress archives                     |
|file                 |Create/Use Buildcaches                           | 
|gnupg2               |Sign/Verify Buildcaches                          |
|git                  |Manage Software Repositories                     |
|svn                  |Manage Software Repositories                     |
|hg                   |Manage Software Repositories                     |
|Python header files  |Bootstrapping from sources                       |

In order to use spack we need to create a **<version>.lua** file under which will point to the spack installation folder and will load the spack environnement.
'vim ${path_of_choice}/modules/default/modulefiles/spack/<version>.lua'

An example of the **<version.lua>** file is shown below 
'''help([[
Spack installation for personal HPC User & Applications Support use.
]])

whatis("Name: Spack")
whatis("Version: <version>")

local activate = [==[
source ${path_to_spack}/share/spack/setup-env.sh
]==]

set_shell_function("spack_activate", activate)
'''
We can now use the following commands to load Spack to our environment 
'module use ${path_of_choice}/modules/default/modulefiles 
spack_activate '

Spack requires a compiler in order to install a package. We can load a compiler of choice using the command **module load <compiler>** and add it to spack using 
'spack comiler find'

Spack searches for available compilers and create a **compiler.yaml** which will be used when installing a package.
Note: you can load multiple compiler and run **spack compiler find**. This will allow the user to choose a specific compiler for his application. We will discuss this later.

We can take advantage of the packages already provided by using the command 
'spack external find'
or if we are searching for a specific package  
'spack external find <package>'

The previous command will create and populate a file **packages.yaml** which will contain the necessary information about the found packages.
The user can also manually write **packages.yaml** and add other package by pointing to their location.




## Creating Environments by Name

To create a basic Conda environment, we'll start by running

`conda create --name mypy python`

where the `--name` option (or the shortened `-n`) means the environment will be specified by **name** and `myenv` will be the name of the created environment.  Any arguments following the environment name are the packages to be installed.

To specify a specific version of a package, simply add the version number after the "=" sign

`conda create --name mypy37 python=3.7`

You can specify multiple packages for installation during environment creation

`conda create --name mynumpy python=3.7 numpy`

Conda ensures dependencies are satisfied when installing packages, so the version of the numpy package installed will be consistent with Python 3.7 (and any other packages specified).

<div class="alert alert-block alert-info">
<br><b>Tip:</b> It’s recommended to install all the packages you want to include in an environment at the same time to help avoid dependency conflicts.
<br><br>
</div>

## Environment Navigation

To see a list of all existing environments (useful to confirm the successful creation of a new environment):

`conda env list`

To activate your new environment:

`conda activate mypy`

Your usual command prompt should now be prefixed with `(mypy)`, which helps keep track of which environment is currently activated.

To see which packages are installed from *within a currently active environment*:

`conda list`

When finished with this programming session, deactivate your environment with:

`conda deactivate`

## Creating Environments by Location

Creating environments by location is especially helpful when working on the Eagle HPC, as the default location is your `/home/<username>/` directory, which is limited to 50 GB.  To create a Conda environment somewhere besides the default location, use the `--prefix` flag (or the shortened `-p`) instead of `--name` when creating.

`conda create --prefix /path/to/mypy python=3.7 numpy`

This re-creates the python+numpy environment from earlier, but with all downloaded packages stored in the specified location.

<div class="alert alert-block alert-danger">
<br>
<b>Warning:</b>  Keep in mind that scratch on Eagle is <b>temporary</b> in that files are purged after 28 days of inactivity.
<br><br>
</div>

Unfortunately, placing environments outside of the default /env folder means that it needs to be activated with the full path (`conda activate /path/to/mypy`) and will show the full path rather than the environment name at the command prompt. 

To fix the cumbersome command prompt, simply modify the `env_prompt` setting in your `.condarc` file:

`conda config --set env_prompt '({name}) '`

Note that `'({name})'` is not a placeholder for your desired environment name but text to be copied literally.  This will edit your `.condarc` file if you already have one or create a `.condarc` file if you do not. For more on modifying your `.condarc` file, check out the [User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html).  Once you've completed this step, the command prompt will show the shortened name (mypy, in the previous example).

## Managing Conda Environments

Over time, it may become necessary to add additional packages to your environments.  New packages can be installed in the currently active environment with:

`conda install pandas`

Conda will ensure that all dependencies are satisfied which may include upgrades to existing packages in this repository.  To install packages from other sources, specify the `channel` option:

`conda install --channel conda-forge fenics`

To add a pip-installable package to your environment:

`
conda install pip
pip <pip_subcommand>
`

<div class="alert alert-block alert-danger">
<br>
<b>A note on mixing Conda and Pip:</b>  Issues may arise when using pip and conda together. When combining conda and pip, it is best to use an isolated conda environment. <b>Only after conda has been used to install as many packages as possible should pip be used to install any remaining software</b>. If modifications are needed to the environment, it is best to create a new environment rather than running conda after pip. When appropriate, conda and pip requirements should be stored in text files.
<br><br>
</div>

For more information on this point, check out the [User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment)

We can use `conda list` to see which packages are currently installed, but for a more version-control-flavored approach:

`conda list --revisions`

which shows changes to the environment over time.  To revert back to a previous environemnt

`conda install --revision 1`


To remove packages from the currently activated environment:

`conda remove pkg1`

To completely remove an environment and all installed packages:

`conda remove --name mypy --all`

Conda environments can become large quickly due to the liberal creation of cached files.  To remove these files and free up space you can use

`conda clean --all`

or to simply preview the potential changes before doing any actual deletion

`conda clean --all --dry-run`

## Sharing Conda Environments

To create a file with the the *exact* "recipe" used to create the current environment:

`conda env export > environment.yaml`

In practice, this recipe may be overly-specific to the point of creating problems on different hardware.  To save an abbreviated version of the recipe with only the packages you *explicitly requested*: 

`conda env export --from-history > environment.yaml`

To create a new environment with the recipe specified in the .yaml file:

`conda env create --name mypyeagle --file environment.yaml`

If a name or prefix isn't specified, the environment will be given the same name as the original environment the recipe was exported from (which may be desirable if you're moving to a different computer).


## Speed up dependency solving

To speed up dependency solving, substitute the mamba command for conda.  Mamba is a dependency solver written in C++ designed to speed up the conda environment solve.

`mamba create --prefix /path/to/mypy python=3.7 numpy`

## Reduce home directory usage

By default, the conda module uses the home directory for package caches and named environments.  This results in a lot of the home directory quota used. Some ways to reduce home directory usage include:

* Use the -p PATH_NAME switch when creating or updating your environment.  Make sure PATH_NAME isn't in the home directory.  Keep in mind files in /scratch are deleted after about a month of inactivity.

* Change the directory used for caching.  This location is set by the module file to ~/.conda-pkgs.  Calling export CONDA_PKGS_DIRS=PATH_NAME to somewhere to store downloads and cached files such as /scratch/$USER/.conda-pkgs will reduce home directory usage.  

## Eagle Considerations

Interacting with your Conda environments on Eagle should feel exactly the same as working on your desktop.  An example desktop-to-HPC workflow might go:

1. Create the environment locally
2. Verify that environment works on a minimal working example
3. Export local environment file and copy to Eagle
4. Duplicate local environment on Eagle
5. Execute production-level runs on Eagle

```bash
#!/bin/bash 
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=5
#SBATCH --account=<project_handle>

module purge
module load conda
conda activate mypy

srun -n 8 python my_main.py
```

## Cheat Sheet of Common Commands

| Task                | ... outside environment                       | ... inside environment                |
|---------------------|-----------------------------------------------|---------------------------------------|
| Create by name      | `conda create -n mypy pkg1 pkg2`              | N/A                                   |
| Create by path      | `conda create -p path/to/mypy pkg1 pkg2`      | N/A                                   |
| Create by file      | `conda env create -f environment.yml`         | N/A                                   |
| Show environments   | `conda env list`                              | N/A                                   |
| Activate            | `conda activate mypy`                         | N/A                                   |
| Deactivate          | N/A                                           | `conda deactivate`                    |
| Install New Package | `conda install -n mypy pkg1 pkg2`             | `conda install pkg1 pkg2`             |
| List All Packages   | `conda list -n mypy`                          | `conda list`                          |
| Revision Listing    | `conda list --revisions -n mypy`              | `conda list --revisions`              |
| Export Environment  | `conda env export -n mypy > environment.yaml` | `conda env export > environment.yaml` |
| Remove Package      | `conda remove -n mypy pkg1 pkg2`              | `conda remove pkg1 pkg2`              |

