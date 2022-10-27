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

<pre>
$ salloc --time=30 --account=<handle> --nodes=2
salloc: Pending job allocation 512998
salloc: job 512998 queued and waiting for resources
salloc: job 512998 has been allocated resources
salloc: Granted job allocation 512998
salloc: Waiting for resource configuration
salloc: Nodes r2i2n5,r2i2n6 are ready for job
[hpc_user@r2i2n5 ~]$ 
</pre>

You can view the nodes that are assigned to your interactive jobs using one of these methods:

<pre>
$ echo $SLURM_NODELIST
r2i2n[5-6]
$ scontrol show hostname
r2i2n5
r2i2n6
</pre>

Once a job is allocated, you will automatically "ssh" to the first allocated node so you do not need to manually ssh to the node after it is assigned. If you requested more than one node, you may ssh to any of the additional nodes assigned to your job. 

You may load modules, run applications, start GUIs, etc., and the commands will execute on that node instead of on the login node.

NOTE: When requesting multiple nodes, please use number of nodes `--nodes` (or `-N`) instead of number of tasks `--ntasks` (or `-n`) to reduce the total number of network "hops" between the allocated nodes.  

Type `exit` when finished using the node.

Interactive jobs are useful for many tasks. For example, to debug a job script, users may submit a request to get a set of nodes for interactive use. When the job starts, the user "lands" on a compute node, with a shell prompt. Users may then run the script to be debugged many times without having to wait in the queue multiple times.

A debug job allows up to two nodes to be available with shorter wait times when the system is heavily utilized. This is accomplished by limiting the number of nodes to 2 per job allocation and specifying `--partition=debug`. For example:

<pre>
[hpc_user@el1 ~]$ salloc --time=60 --account=<handle> --nodes=2 --partition=debug
</pre>

A debug node will only be available for a maximum wall time of 1 hour.

## Sample Interactive Job Commands

The following command requests interactive access to one node with at least 150 GB RAM for 20 minutes:

<pre>
$ salloc --time=20 --account=<handle> --nodes=1 --mem=150G
</pre>

For an interactive job that will require multiple nodes, for example, running interactive software that uses MPI, launch with an salloc first:

<pre>
$ salloc --time=20 --account=<handle> --nodes=2
</pre>

The above salloc command will log you into one of the two nodes automatically. You can then launch your software using an srun command with the appropriate flags, such as --ntasks or --ntasks-per-node:

<pre>
[hpc_user@r2i2n5 ~]$ module purge; module load paraview
[hpc_user@r2i2n5 ~]$ srun --ntasks=20 --ntasks-per-node=10 pvserver --force-offscreen-rendering
</pre>

If your single-node job needs a GUI that uses X-windows:

<pre>
$ ssh -Y kestrel.hpc.nrel.gov
...
$ salloc --time=20 --account=<handle> --nodes=1 --x11
</pre>

If your multi-node job needs a GUI that uses X-windows, the least fragile mechanism is to acquire nodes as above, then in a separate session set up X11 forwarding:

<pre>
$ salloc --time=20 --account=<handle> --nodes=2
...
[hpc_user@r3i5n13 ~]$ (your compute node r3i5n13)
</pre>

Then from your local workstation:

<pre>
$ ssh -Y kestrel.hpc.nrel.gov
...
[hpc_user@el1 ~]$ ssh -Y r3i5n13  #(from login node to reserved compute node)
...
[hpc_user@r3i5n13 ~]$  #(your compute node r3i5n13, now X11-capable)
[hpc_user@r3i5n13 ~]$ xterm  #(or another X11 GUI application)
</pre>

## Requesting Interactive GPU Nodes

The following command requests interactive access to GPU nodes:

<pre>
[hpc_user@el2 ~] $ salloc --account=<handle> --time=5 --gres=gpu:2 
</pre>

This next srun command inside the interactive session gives you access to the GPU devices:

<pre>
[hpc_user@r104u33 ~] $ srun --gres=gpu:2 nvidia-smi
Mon Oct 21 09:03:29 2019
+-------------------------------------------------------------------+
| NVIDIA-SMI 410.72 Driver Version: 410.72 CUDA Version: 10.0 |
|---------------------+----------------------+----------------------+
| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. |
|=====================+======================+======================|
| 0 Tesla H100-PCIE... Off | 00000000:37:00.0 Off | 0 |
| N/A 41C P0 38W / 250W | 0MiB / 16130MiB | 0% Default |
+---------------------+----------------------+----------------------+
| 1 Tesla H100-PCIE... Off | 00000000:86:00.0 Off | 0 |
| N/A 40C P0 36W / 250W | 0MiB / 16130MiB | 0% Default |
+---------------------+----------------------+----------------------+

+-------------------------------------------------------------------+
| Processes: GPU Memory |
| GPU PID Type Process name Usage |
|===================================================================|
| No running processes found |
+-------------------------------------------------------------------+
</pre>

