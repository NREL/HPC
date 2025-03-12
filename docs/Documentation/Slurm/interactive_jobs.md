---
layout: default
title: Running Interactive Jobs
has_children: false
---

# Running Interactive Jobs

Interactive jobs provide a shell prompt on a compute node. This allows users to execute commands and scripts "live" as they would on the login nodes, with direct user input and output immediately available. 

Login nodes are primarily intended to be used for logging in, editing scripts, and submitting batch jobs. Interactive work that involves substantial resources—either memory, CPU cycles, or file system I/O—should be performed on the compute nodes rather than on login nodes.

Interactive jobs may be submitted to any partition and are subject to the same time and node limits as non-interactive jobs.

## Requesting Interactive Access

The `salloc` command is used to start an interactive session on one or more compute nodes. When resources become available, interactive access is provided by a shell prompt. The user may then work interactively on the node for the time specified.

The job is held until the scheduler can allocate a node to you. You will see a series of messages such as: 

```
$ salloc --time=30 --account=<handle> --nodes=2
salloc: Pending job allocation 512998
salloc: job 512998 queued and waiting for resources
salloc: job 512998 has been allocated resources
salloc: Granted job allocation 512998
salloc: Waiting for resource configuration
salloc: Nodes x1008c7s6b1n0,x1008c7s6b1n1 are ready for job
[hpc_user@x1008c7s6b1n0 ~]$ 
```

You can view the nodes that are assigned to your interactive jobs using one of these methods:

```
$ echo $SLURM_NODELIST
x1008c7s6b1n[0-1]
$ scontrol show hostname
x1008c7s6b1n0
x1008c7s6b1n1
```

Once a job is allocated, you will automatically "ssh" to the first allocated node so you do not need to manually ssh to the node after it is assigned. If you requested more than one node, you may ssh to any of the additional nodes assigned to your job. 

You may load modules, run applications, start GUIs, etc., and the commands will execute on that node instead of on the login node.

!!! note
    When requesting multiple nodes, please use number of nodes `--nodes` (or `-N`) instead of number of tasks `--ntasks` (or `-n`) to reduce the total number of network "hops" between the allocated nodes.  

Type `exit` when finished using the node.

Interactive jobs are useful for many tasks. For example, to debug a job script, users may submit a request to get a set of nodes for interactive use. When the job starts, the user "lands" on a compute node, with a shell prompt. Users may then run the script to be debugged many times without having to wait in the queue multiple times.

A debug job allows up to two nodes to be available with shorter wait times when the system is heavily utilized. This is accomplished by specifying `--partition=debug`. For example:

```
[hpc_user@kl1 ~]$ salloc --time=60 --accounft=<handle> --partition=debug
```

Add `--nodes=2` to claim two nodes.

Add `--gpus=#` (substituting the number of GPUs you want to use) to claim a debug GPU node. Note that there are fewer GPU nodes in the debug queue, so there may be more of a wait time.

A debug job on any node type will only be available for jobs with a maximum walltime (--time) of 1 hour, and only one debug job at a time is permitted per person.

## Sample Interactive Job Commands

The following command requests interactive access to one node with at least 150 GB RAM for 20 minutes:

```
$ salloc --time=20 --account=<handle> --nodes=1 --mem=150G
```

For an interactive job that will require multiple nodes, for example, running interactive software that uses MPI, launch with an salloc first:

```
$ salloc --time=20 --account=<handle> --nodes=2
```

The above salloc command will log you into one of the two nodes automatically. You can then launch your software using an srun command with the appropriate flags, such as --ntasks or --ntasks-per-node:

```
[hpc_user@x1008c7s6b1n0 ~]$ module purge; module load paraview
[hpc_user@x1008c7s6b1n0 ~]$ srun --ntasks=20 --ntasks-per-node=10 pvserver --force-offscreen-rendering
```

If your single-node job needs a GUI that uses X-windows:

```
$ ssh -Y kestrel.hpc.nrel.gov
...
$ salloc --time=20 --account=<handle> --nodes=1 --x11
```

If your multi-node job needs a GUI that uses X-windows, the least fragile mechanism is to acquire nodes as above, then in a separate session set up X11 forwarding:

```
$ salloc --time=20 --account=<handle> --nodes=2
...
[hpc_user@x1008c7s6b1n0 ~]$ (your compute node x1008c7s6b1n0)
```

Then from your local workstation:

```
$ ssh -Y kestrel.hpc.nrel.gov
...
[hpc_user@kl1 ~]$ ssh -Y x1008c7s6b1n0  #(from login node to reserved compute node)
...
[hpc_user@x1008c7s6b1n0 ~]$  #(your compute node x1008c7s6b1n0, now X11-capable)
[hpc_user@x1008c7s6b1n0 ~]$ xterm  #(or another X11 GUI application)
```

From a Kestrel-DAV Fastx remote desktop session, you can omit the `ssh -Y kestrel.hpc.nrel.gov` above since your terminal in FastX will already be connected to a DAV (kd#) login node. 


## Requesting Interactive GPU Nodes

The following command requests interactive access to GPU nodes:

```
[hpc_user@kl2 ~] $ salloc --account=<handle> --time=5 --gpus=2
```
You may run the nvidia-smi command to confirm the GPUs are visible:

```
[hpc_user@x3100c0s29b0n0 ~] $ nvidia-smi
Wed Mar 12 16:20:53 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:04:00.0 Off |                    0 |
| N/A   40C    P0             71W /  699W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:64:00.0 Off |                    0 |
| N/A   40C    P0             73W /  699W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```
