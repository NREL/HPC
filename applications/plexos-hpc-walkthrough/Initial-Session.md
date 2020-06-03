## Initial Session

A [getting started guide](https://hpc.nrel.gov/users/systems/peregrine/getting-started-for-users-new-to-high-performance-computing) is available on the [NREL HPC Website](https://hpc.nrel.gov)

Here we walk through an initial session on the HPC, going to the scratch filesystem, and obtaining an interactive session on a compute node, finding the plexos software.

```bash
[wjones@login4 ~]$ pwd
/home/wjones
```
On the HPC system we will work out of the scratch filesystem.  You can put files you want to keep for a long time in the home filesystem, but it is much smaller than the scratch filesystem and we will not want to run large compute and data intensive jobs from that filesystem.

```bash
[wjones@login4 scratch]$ cd /scratch/$USER
[wjones@login4 wjones]$ pwd
/scratch/wjones
```

We will aquire an interactive session on a compute node to do our work and will request it from the batch scheduler using qsub.
```
[wjones@login4 wjones]$ qsub -I -A PLEXOSMODEL -l advres=workshop.57721,nodes=1,walltime=30:00 -q batch-h 
qsub: waiting for job 3211739 to start
qsub: job 3211739 ready

[wjones@n0289 wjones]$ qstat -u $USER

hpc-admin2.hpc.nrel.gov: 
                                                                                  Req'd       Req'd       Elap
Job ID                  Username    Queue    Jobname          SessID  NDS   TSK   Memory      Time    S   Time
----------------------- ----------- -------- ---------------- ------ ----- ------ --------- --------- - ---------
3211739                 wjones      debug    STDIN             60456     1      1       --   00:30:00 R  00:00:09
[wjones@n0289 wjones]$ pwd
/scratch/wjones
```
We landed in the scratch file system on the new host, n0289, and can see that we have one job running using qstat.

'modules' is used to manage software that we have available to execute from the command line.  Here, we 
list our currently loaded software, 
purge our currently loaded software,
make available the plexos software, 
load the plexos software which depends on mono and the xpressmp solvers.

```bash
[wjones@n0289 wjones]$ module list
Currently Loaded Modulefiles:
  1) comp-intel/13.1.3         2) impi-intel/4.1.1-13.1.3
[wjones@n0289 wjones]$ module use /nopt/nrel/apps/modules/candidate/modulefiles
[wjones@n0289 wjones]$ module avail plexos

------------------------------------------------ /nopt/nrel/apps/modules/candidate/modulefiles -------------------------------------------------
plexos/6.400.2 plexos/7.200.2 plexos/7.300.3 plexos/7.300.4 plexos/7.400.2
[wjones@n0289 wjones]$ module purge
[wjones@n0289 wjones]$ module load plexos/7.400.2
plexos/7.400.2(16):ERROR:151: Module 'plexos/7.400.2' depends on one of the module(s) 'mono/4.6.2.7'
plexos/7.400.2(16):ERROR:102: Tcl command execution failed: prereq mono/4.6.2.7

[wjones@n0289 wjones]$ module load mono/4.6.2.7
[wjones@n0289 wjones]$ module load xpressmp/8.0.4
[wjones@n0289 wjones]$ module load plexos/7.400.2
[wjones@n0289 wjones]$ module list
Currently Loaded Modulefiles:
  1) mono/4.6.2.7     2) xpressmp/8.0.4   3) plexos/7.400.2
```
