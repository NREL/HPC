---
title: July 2021 NREL HPC Monthly Update
data: 2021-07-06
layout: default
brief: CSC User & Applications Support, Eagle Job Queue, Eagle System Time
---

# CSC User & Applications Support
We will be making the conda/4.9.2 module the default module for loading (i.e., without a version number specified). By way of reminders,

* If you need to reference the existing default Anaconda installation, add the version number to your module load statement in job scripts, 
i.e., `module load conda/mini_py37_4.8.3` rather than just `module load conda`. Custom environments should interoperate with either version, though.
* This module permits `conda activate` and `conda deactivate` functionality without conda init. Don't use conda init, as it breaks login shell setup.
* Consider trying mamba instead of conda when setting up environments. For example, `mamba install` to add a package to a custom environment.

Julia modules are now available on Eagle. The module files are available at /nopt/nrel/ecom/modulefiles. Currently, versions 1.5.4 and 1.6.1 are available.

If you plan on using these module files regularly, you may wish to add this directory to your module search path with the command:
`module use -a /nopt/nrel/ecom/modulefiles`

You can add this command to your .bash_profile or .bashrc file with the following command:
`echo 'module use -a /nopt/nrel/ecom/modulefiles' >> .bash_profile`

(or .bashrc in place of .bash_profile).

Once your module path is updated, simply load the desired Julia version module:
`module load julia`

Questions or problems regarding Julia on Eagle can be sent to <jonathan.maack@nrel.gov>.

# Eagle Job Queue
The number of running jobs on Eagle has been dipping on the weekends. Please think about submitting jobs to run over the weekend, 
especially long weekends, so we can keep the system full.

# Eagle System Time
The next Eagle system time is scheduled for the week of August 2nd. This will be a multi day outage to do updates to the parallel 
file system as well as take care of some hardware issues affecting the compute nodes.  Eagle and related file systems will be unavailable at during this system time. 
