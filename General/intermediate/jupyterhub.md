---
layout: default
title: JupyterHub
grand_parent: General
parent: Intermediate
---

# JupyterHub
Prior to using Jupyterhub, you will have had to have logged into Eagle via the command line at least once.

Given that, to start using Jupyterhub on Eagle, go to [Europa](https://europa.hpc.nrel.gov) in your local machine's browser, and log in with your Eagle username and password. 
You should land in your home directory, and see everything there via the standard Jupyter file listing.

From the "New" pulldown on the right hand side, you can start a notebook, open a terminal, or create a file or folder. 
The default installation is Python version 3, and a variety of Conda modules are installed already. You can start a
Python3 notebook right away, and access the Python modules that are already present. To see what's installed, from a notebook
you can use the following command:

```
!conda list
```

Alternatively, you can start a Terminal, and use the usual conda commands from the shell.

## Creating a custom environment to access from the notebook

Start a Terminal session, and follow the instructions on the [HPC website](https://www.nrel.gov/hpc/eagle-software-python.html) 
to create an environment. Now, to make this environment visible from your future notebooks, run the following command:

```
source activate <myenv>
python -m ipykernel install --user --name <myenv> --display-name "How-you-want-your-custom-kernel-to-appear-in-the-notebook-pulldown (<myenv>)"
```

where `<myenv>` is the argument to `-n` you used in your `conda create` command.

After running this command, when you open a new notebook, you should see as an option your new environment, and once loaded
be able to access all Python modules therein.

## Using Jupyterhub from Eagle

To use inside Eagle, the Jupyterhub server exists on the internal network @ https://europa-int/.
