---
layout: default
title: Conda
parent: Python
grand_parent: Programming Languages
---

![conda logo]({{site.baseurl}}/assets/conda_logo.png)

## Why Conda?

Conda is a **package manager** which allows you to easily create and switch betwen different software environments in different languages for different purposes.  With Conda, it's easy to:

* Manage different (potentially conflicting) versions of the same software without complication

* Quickly stand up even complicated dependencies for stacks of software

* Share your specific programming environment with others for reproducible results 


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
modlue load conda
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

