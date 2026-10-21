# Environment Comparisons

If you work with Linux long enough you'll someday run into "That worked yesterday but not today" or "That works for her and not for me."

In the first case either something changed in your user space or some installed program changed.

In the second case there might be slight differences in the setup of your environments. 

The python program picenv is designed to make it easy to track down differences in environments; that is the output from printenv. 

Two environment to be compared are "archived" with picenv.  Then picenv is run in a compare mode and it prints three concise lists of differences:
1.  Variables in both environments but differ
2.  Variables only in the first environment
3.  Variables only in the second environment
 
This is much easier than sorting through the normal output from printenv.  

There is a verbose mode which prints addition details.

There is also a mode for comparing module avail output.  However, loaded modules and their variables are contained in the "normal" archive.   


The program has a unique feature.  If it is instead named picmod it will by default look at modules instead of the environment; that is module avail.

If you copy picenv to some directory you can create an alias picmod to have the same program work on both environments and modules.  You create an alias like this:

```
[tkaiser2@gila-compute-1 bin]$ ls picenv
picenv
[tkaiser2@gila-compute-1 bin]$ ln -s picenv picmod
[tkaiser2@gila-compute-1 bin]$ ls -l pic*
-rwx--xr-- 1 tkaiser2 tkaiser2 5728 Aug 10 12:53 picenv
lrwxrwxrwx 1 tkaiser2 tkaiser2    6 Aug 10 13:01 picmod -> picenv
[tkaiser2@gila-compute-1 bin]$ 
```
Here is "help" from the program

```
[tkaiser2@gila-compute-1 bin]$ picenv -help
Usage:
picenv | picmod  outfile
     create a python pickle file containing either the module 
avail list or the linux environment. 

     The default output is dependent on the name of this program.  If it 
contains `env` the output will be for the environment.  If it contains `mod`
it will be a dump of `module avail`.  The command line options `-e` and `-m` 
override the program name convention.

picenv | picmod  infile1 infile2 [-v]
     compare two files.  -v = show detailed differences

picenv | picmod  infile1 -d
     dump the contents of the file as text

picenv | picmod  infile1 -M
     created a python pickle file from the input text file from module avail
     input file is required

picenv | picmod  infile1 -E
     created a python pickle file from the input text file from printenv
     input file is required
[tkaiser2@gila-compute-1 bin]$ 
```

Without any inputs:
* picenv - creates a file *env_dump* which is a python pickle (binary) file containing the current environment
* picmod - creates a file *mod_dump* which is a python pickle (binary) file containing the **currently available modules**.  

Note: picenv will grab the loaded modules so we don't have a separate command for this function. 
    
The pickle file is not human readable.  However, you can dump the contents of the file in text using the -d option.  For the environment the output is easier to read than from printenv.  We can also look for particular variables.  For example to get the loaded modules

```
[tkaiser2@gila-compute-1 bin]$ picenv -d  env_dump | grep LOADEDMODULES -A 1
b'LOADEDMODULES' :
 b'Core/25.05:DefApps' 
```

    
Why create a non-human readable file when the started with with human readable input?  It is difficult to compare long unsorted text files containing the description of your environment.   The main usage of picenv is to save environment for comparison.  Just doing printenv creates a record of your current environment that can easily be compared to another.

Here is an example.  We start with a base environment, run picenv, then load a module, run picenv again, and finally run picenv to compare the environments.

```

[tkaiser2@kl3 examples]$picenv base
[tkaiser2@kl3 examples]$ml gcc
[tkaiser2@kl3 examples]$picenv mlgcc
[tkaiser2@kl3 examples]$pidenv base mlgcc
-bash: pidenv: command not found
[tkaiser2@kl3 examples]$picenv base mlgcc
**** In both but differ:
b'LD_LIBRARY_PATH'
b'LOADEDMODULES'
b'MANPATH'
b'MODULEPATH'
b'PATH'
b'_LMFILES_'
b'_ModuleTable001_'
b'_ModuleTable002_'
b'_ModuleTable003_'
b'_ModuleTable_Sz_'
b'__LMOD_Priority_MODULEPATH'
b'__LMOD_REF_COUNT_LD_LIBRARY_PATH'
b'__LMOD_REF_COUNT_MANPATH'
b'__LMOD_REF_COUNT_MODULEPATH'
b'__LMOD_REF_COUNT_PATH'

**** Only in  mlgcc :
b'CPLUS_INCLUDE_PATH'
b'C_INCLUDE_PATH'
b'LIBRARY_PATH'
b'LMOD_FAMILY_COMPILER'
b'LMOD_FAMILY_COMPILER_VERSION'
b'NREL_FAMILY_COMPILER'
b'NREL_FAMILY_COMPILER_VERSION'
b'NREL_GCC_ROOT'
b'_ModuleTable004_'
b'__LMOD_REF_COUNT_CPLUS_INCLUDE_PATH'
b'__LMOD_REF_COUNT_C_INCLUDE_PATH'
b'__LMOD_REF_COUNT_LIBRARY_PATH'

**** Only in  base :
[tkaiser2@kl3 examples]$
```
We see that there are several changes.  Running with the -v (verbose) flag will show the differences.  We can use grep to pull out specific variables.

```

[tkaiser2@kl3 examples]$picenv base mlgcc -v | grep LD_LIBRARY_PATH -A2
b'LD_LIBRARY_PATH'
base b'/nopt/slurm/latest/lib::'
mlgcc b'/nopt/nlr/apps/kestrel-cpu/gcc/14.2.0/lib64:/nopt/slurm/latest/lib::'
```
We see that loading the module added /nopt/nlr/apps/kestrel-cpu/gcc/14.2.0/lib64 to LD_LIBRARY_PATH.

If you run using a batch scheduler such as slurm you can see the differences in environments from the login node, and interactive session, and a batch session.  

We ran 
On the login node we ran
```
picenv envlogin
```
In and interactive session on a compute node we ran
```
picenv envinteractive
```

Finally we ran the following batch script

```
#!/bin/bash
#SBATCH --job-name="install"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=60
#SBATCH --partition=amd
#SBATCH --account=hpcapps

~/bin/picenv envbatch
```
Then we can compare the environments from the interactive and batch sessions.
```
[tkaiser2@gila-compute-8 ~]$ picenv envinteractive envbatch
**** In both but differ:
b'HISTFILE'
b'HOSTNAME'
b'LD_LIBRARY_PATH'
b'MODULEPATH'
b'PATH'
b'SLURM_JOBID'
b'SLURM_JOB_END_TIME'
b'SLURM_JOB_ID'
b'SLURM_JOB_NAME'
b'SLURM_JOB_START_TIME'
b'SLURM_TASK_PID'
b'TMPDIR'
b'_ModuleTable002_'
b'_ModuleTable_Sz_'
b'__LMOD_REF_COUNT_MODULEPATH'

**** Only in  envbatch :
b'ENVIRONMENT'
b'_ModuleTable003_'

**** Only in  envinteractive :
b'SLURMD_DEBUG'
b'SLURM_JOB_GROUP'
b'SLURM_LAUNCH_NODE_IPADDR'
b'SLURM_MPI_TYPE'
b'SLURM_PMI2_PROC_MAPPING'
b'SLURM_PMI2_SRUN_PORT'
b'SLURM_PMI2_STEP_NODES'
b'SLURM_PTY_PORT'
b'SLURM_PTY_WIN_COL'
b'SLURM_PTY_WIN_ROW'
b'SLURM_SRUN_COMM_HOST'
b'SLURM_SRUN_COMM_PORT'
b'SLURM_STEPID'
b'SLURM_STEP_ID'
b'SLURM_STEP_LAUNCHER_PORT'
b'SLURM_STEP_NODELIST'
b'SLURM_STEP_NUM_NODES'
b'SLURM_STEP_NUM_TASKS'
b'SLURM_STEP_TASKS_PER_NODE'
b'SRUN_DEBUG'
[tkaiser2@gila-compute-8 ~]$ 





[tkaiser2@gila-compute-8 ~]$ picenv envinteractive envbatch -v | grep LD_LIBRARY_PATH -A2
b'LD_LIBRARY_PATH'
envinteractive b'/nopt/slurm/current/x86_64/lib:/nopt/slurm/current/x86_64/lib:'
envbatch b'/nopt/slurm/current/x86_64/lib:'
[tkaiser2@gila-compute-8 ~]$ 




[tkaiser2@gila-compute-8 ~]$ picenv envinteractive envlogin
**** In both but differ:
b'HISTFILE'
b'LD_LIBRARY_PATH'
b'MODULEPATH'
b'PATH'
b'SHLVL'
b'_ModuleTable002_'
b'_ModuleTable_Sz_'
b'__LMOD_REF_COUNT_MODULEPATH'

**** Only in  envlogin :
b'_ModuleTable003_'

**** Only in  envinteractive :
b'HYDRA_BOOTSTRAP'
b'HYDRA_LAUNCHER_EXTRA_ARGS'
b'I_MPI_HYDRA_BOOTSTRAP'
b'I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS'
b'OMPI_MCA_plm_slurm_args'
b'PRTE_MCA_plm_slurm_args'
b'SLURMD_DEBUG'
b'SLURMD_NODENAME'
b'SLURM_CLUSTER_NAME'
b'SLURM_CPUS_ON_NODE'
b'SLURM_GTIDS'
b'SLURM_JOBID'
b'SLURM_JOB_ACCOUNT'
b'SLURM_JOB_CPUS_PER_NODE'
b'SLURM_JOB_END_TIME'
b'SLURM_JOB_GID'
b'SLURM_JOB_GROUP'
b'SLURM_JOB_ID'
b'SLURM_JOB_NAME'
b'SLURM_JOB_NODELIST'
b'SLURM_JOB_NUM_NODES'
b'SLURM_JOB_PARTITION'
b'SLURM_JOB_QOS'
b'SLURM_JOB_START_TIME'
b'SLURM_JOB_UID'
b'SLURM_JOB_USER'
b'SLURM_LAUNCH_NODE_IPADDR'
b'SLURM_LOCALID'
b'SLURM_MPI_TYPE'
b'SLURM_NNODES'
b'SLURM_NODEID'
b'SLURM_NODELIST'
b'SLURM_NPROCS'
b'SLURM_NTASKS'
b'SLURM_NTASKS_PER_NODE'
b'SLURM_OOM_KILL_STEP'
b'SLURM_PMI2_PROC_MAPPING'
b'SLURM_PMI2_SRUN_PORT'
b'SLURM_PMI2_STEP_NODES'
b'SLURM_PRIO_PROCESS'
b'SLURM_PROCID'
b'SLURM_PTY_PORT'
b'SLURM_PTY_WIN_COL'
b'SLURM_PTY_WIN_ROW'
b'SLURM_SCRIPT_CONTEXT'
b'SLURM_SRUN_COMM_HOST'
b'SLURM_SRUN_COMM_PORT'
b'SLURM_STEPID'
b'SLURM_STEP_ID'
b'SLURM_STEP_LAUNCHER_PORT'
b'SLURM_STEP_NODELIST'
b'SLURM_STEP_NUM_NODES'
b'SLURM_STEP_NUM_TASKS'
b'SLURM_STEP_TASKS_PER_NODE'
b'SLURM_SUBMIT_DIR'
b'SLURM_SUBMIT_HOST'
b'SLURM_TASKS_PER_NODE'
b'SLURM_TASK_PID'
b'SLURM_TOPOLOGY_ADDR'
b'SLURM_TOPOLOGY_ADDR_PATTERN'
b'SRUN_DEBUG'
b'TMPDIR'
[tkaiser2@gila-compute-8 ~]$ 
```