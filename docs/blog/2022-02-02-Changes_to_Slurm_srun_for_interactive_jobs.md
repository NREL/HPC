# Changes to Slurm "srun" for Interactive Jobs

The Slurm job scheduler was upgraded during the recent system time on January 10, 2022. 
One of the side effects of this was a change in the way Slurm handles job steps internally in certain cases. 
This may affect the way some users run job steps with srun inside of interactive jobs `srun --pty $SHELL`, so we wanted to provide some guidance as we work on updating our documentation to reflect this change. 

When running an interactive job with `srun --pty $SHELL` and then launching job steps on a node, a second `srun` is often used "inside" the first srun to launch certain software. 
For example, for users of Paraview, a Paraview server may be launched on an interactive node with `srun -n 8 pvserver --force-offscreen-rendering`. 
(Certain GPU-enabled or MPI-enabled interactive software also functions in a similar manner.)

This "srun-inside-an-srun" process will no longer function in the same way as in the past. Instead, the "outer" `srun` should be replaced with an `salloc` command. `salloc` will accept the same arguments as srun, but `--pty $SHELL` will no longer be required. `salloc` will automatically open a shell to the node once the job starts, and the "inner" `srun` can then be run successfully as normal.

Other regular uses of srun and srun inside sbatch scripts should continue to behave as expected. 

For further technical details on this Slurm change, please see the [Slurm 20.11 Release Notes](https://github.com/SchedMD/slurm/blob/slurm-20.11/RELEASE_NOTES) regarding job steps, `srun`, and the new `--overlap` flag.