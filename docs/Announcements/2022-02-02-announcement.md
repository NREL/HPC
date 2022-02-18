---
title: February 2022 Monthly Update
data: 2022-02-02
layout: default
brief: Slurm Change, April Outage
---
# Changes to Slurm "srun" for Interactive Jobs
During the recent system time the Slurm job scheduler was upgraded. One of the side effects of this was a change in the way Slurm handles job steps internally in certain cases. This may affect the way some users run job steps with srun inside of interactive jobs (srun --pty $SHELL), so we wanted to provide some guidance as we work on updating our documentation to reflect this change. 

When running an interactive job with srun --pty $SHELL and then launching job steps on a node, a second srun is often used "inside" the first srun to launch certain software. For example, for users of Paraview, a Paraview server may be launched on an interactive node with "srun -n 8 pvserver --force-offscreen-rendering". (Certain GPU-enabled or MPI-enabled interactive software also functions in a similar manner.)

This "srun-inside-an-srun" process will no longer function in the same way as in the past. Instead, the "outer" srun should be replaced with an salloc command. Salloc will accept the same arguments as srun, but "--pty $SHELL" will no longer be required. Salloc will automatically open a shell to the node once the job starts, and the "inner" srun can then be run successfully as normal.

Other regular uses of srun and srun inside sbatch scripts should continue to behave as expected. 

For further technical details on this Slurm change, please see the [Slurm 20.11 Release Notes](https://github.com/SchedMD/slurm/blob/slurm-20.11/RELEASE_NOTES) regarding job steps, srun, and the new --overlap flag.

# Upcoming Outage
There is a whole campus power outage planned for April 1 for NREL's South Table Mesa (STM) campus. All Computational Science Center systems will be affected. More details will follow as the date approaches.
