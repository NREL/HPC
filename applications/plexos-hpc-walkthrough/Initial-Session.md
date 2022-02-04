## Initial Session

A [getting started guide](https://www.nrel.gov/hpc/eagle-user-basics.html) is available on the [NREL HPC Website](https://www.nrel.gov/hpc)

Here we walk through an initial session on the HPC, going to the scratch filesystem, and obtaining an interactive session on a compute node, finding the plexos software.

```bash
[wjones@el3 ~]$ pwd
/home/wjones
```
On the HPC system we will work out of the scratch filesystem.  You can put files you want to keep for a long time in the home filesystem, but it is much smaller than the scratch filesystem and we will not want to run large compute and data intensive jobs from that filesystem.

```bash
[wjones@el3 scratch]$ cd /scratch/$USER
[wjones@el3 wjones]$ pwd
/scratch/wjones
```

We will aquire an interactive session on a compute node to do our work and will request it from the batch scheduler using salloc.
```
[wjones@el3 wjones]$ salloc -N 1 -t 60 -A hpcapps -p debug
salloc: Pending job allocation 5758298
salloc: job 5758298 queued and waiting for resources
salloc: job 5758298 has been allocated resources
salloc: Granted job allocation 5758298
salloc: Waiting for resource configuration
salloc: Nodes r3i7n35 are ready for job
[wjones@r3i7n35 wjones]$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5758298     debug       sh   wjones  R       0:55      1 r3i7n35
[wjones@r3i7n35 wjones]$ pwd
/scratch/wjones
```
We landed in the scratch file system on the new host, r3i7n35, and can see that we have one job running using squeue.

'modules' is used to manage software that we have available to execute from the command line.  Here, we
list our currently loaded software,
purge our currently loaded software,
make available the plexos software,
load the plexos software which depends on mono and the xpressmp solvers.

```bash
[wjones@r3i7n35 wjones]$ module list
No modules loaded
[wjones@r3i7n35 wjones]$ module avail plexos

--------------------------- /nopt/nrel/apps/modules/default/modulefiles ----------------------------
   plexos/7.300.4        plexos/7.500.2     plexos/8.0         plexos/8.200R01
   plexos/7.400.2 (D)    plexos/8.000R03    plexos/8.100R02

  Where:
   D:  Default Module

Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


[wjones@r3i7n35 wjones]$ module load mono/4.6.2.7
[wjones@r3i7n35 wjones]$ module load xpressmp/8.0.4
[wjones@r3i7n35 wjones]$ module load centos
[wjones@r3i7n35 wjones]$ module load plexos/7.400.2
[wjones@r3i7n35 wjones]$ module list

Currently Loaded Modules:
  1) mono/4.6.2.7   2) xpressmp/8.0.4   3) centos/7.7   4) plexos/7.400.2



```

Please note that, depending on when you are using this resource, the version numbers
of the software might be different. Detailed instructions for the latest available
versions should be available at `HPC/applications/plexos-quick-start/`
