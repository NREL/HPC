# Commands to Monitor and Control Jobs
*Learn about a variety of Slurm commands to monitor and control jobs.*

Please see ```man``` pages for more information on the commands listed below. Also see ```--help``` or ```--usage```.

Also see our [Presentation on Advanced Slurm Features](https://www.nrel.gov/hpc/assets/pdfs/slurm-advanced-topics.pdf), which has supplementary information on how to manage jobs.

On Github, see [another great resource for Slurm on Eagle](https://github.com/sayerhs/nrel-eagle/blob/master/nrel-eagle.md).

| Command | Description |
| ----------| ------------ | 
| ```squeue``` | Show the Slurm queue. Users can specify JOBID or USER.|
| ```scontrol``` | 	Controls various aspects of jobs such as job suspension, re-queuing or resuming jobs and can display diagnostic info about each job.|
| ```scancel``` | Cancel specified job(s). |
| ```sinfo``` | View information about all Slurm nodes and partitions. |
| ```sacct``` | Detailed information on accounting for all jobs and job steps. |
| ```sprio``` | View priority and the factors that determine scheduling priority. |

## Usage Examples

### squeue
```
$ squeue -u hpcuser
           JOBID    PARTITION       NAME      USER   ST       TIME      NODES   NODELIST(REASON)
          506955          gpu   wait_tes   hpcuser   PD       0:00          1      (Resources)
```

```
$ squeue -l
Thu Dec 13 12:17:31 2018
 JOBID  PARTITION NAME     USER     STATE   TIME    TIME_LIMI   NODES  NODELIST(REASON)
 516890 standard Job007    user1    PENDING 0:00    12:00:00    1050   (Dependency)
 516891 standard Job008    user1    PENDING 0:00    12:00:00    1050   (Dependency)
 516897      gpu Job009    user2    PENDING 0:00    04:00:00       1   (Resources)
 516898 standard Job010    user3    PENDING 0:00    15:00:00      71   (Priority)
 516899 standard Job011    user3    PENDING 0:00    15:00:00      71   (Priority)
-----------------------------------------------------------------------------
 516704 standard Job001    user4    RUNNING 4:09:48 15:00:00      71    r1i0n[0-35],r1i1n[0-34]
 516702 standard Job002    user4    RUNNING 4:16:50 15:00:00      71    r1i6n35,r1i7n[0-35],r2i0n[0-33]
 516703 standard Job003    user4    RUNNING 4:16:57 15:00:00      71    r1i5n[0-35],r1i6n[0-34]
 516893 standard Job004    user4    RUNNING 7:19     3:00:00      71    r1i1n35,r1i2n[0-35],r1i3n[0-33]
 516894 standard Job005    user4    RUNNING 7:19     3:00:00      71    r4i2n[20-25],r6i6n[7-35],r6i7n[0-35]
 516895 standard Job006    user4    RUNNING 7:19     3:00:00      71    r4i2n[29-35],r4i3n[0-35],r4i4n[0-20]

```

To estimate when your jobs will start to run, use the ```squeue --start``` command with the JOBID.

```

$ squeue --start -j 509851,509852
 JOBID    PARTITION    NAME      USER      ST          START_TIME    NODES   SCHEDNODES   NODELIST(REASON)
 509851   short      test1.sh   hpcuser    PD                 N/A      100       (null)       (Dependency)
 509852   short      test2.sh   hpcuser    PD 2018-12-19T16:54:00        1      r1i6n35         (Priority)
 
```
### scontrol
To get detailed information about your job before and while it runs, you may use scontrol show job with the JOBID.  For example:
```
$ scontrol show job 522616
JobId=522616 JobName=myscript.sh
 UserId=hpcuser(123456) GroupId=hpcuser(123456) MCS_label=N/A
 Priority=43295364 Nice=0 Account=csc000 QOS=normal
 JobState=PENDING Reason=Dependency Dependency=afterany:522615
```
The ```scontrol``` command can also be used to modify pending and running jobs:
```
$ scontrol update jobid=526501 qos=high
$ sacct -j 526501 --format=jobid,partition,state,qos
       JobID  Partition      State        QOS
------------ ---------- ---------- ----------
526501            short    RUNNING       high
526501.exte+               RUNNING
526501.0                 COMPLETED
```
To pause a job: ```scontrol hold <JOBID>```

To resume a job: ```scontrol resume <JOBID>```

To cancel and rerun: ```scontrol requeue <JOBID>```

### scancel 
Use ```scancel -i <jobID>``` for an interactive mode to confirm each job_id.step_id before performing the cancel operation. Use ```scancel --state=PENDING,RUNNING,SUSPENDED -u <userid>``` to cancel your jobs by STATE or ```scancel -u <userid>``` to cancel ALL of your jobs.

### sinfo
Use ```sinfo``` to view cluster information:
```
$ sinfo -o %A
NODES(A/I)
1580/514
```
Above, ```sinfo``` shows nodes Allocated (A) and nodes idle (I) in the entire cluster.

To see specific node information use ```sinfo -n <node id>``` to show information about a single or list of nodes. You will see the partition to which the node can allocate as well as the node STATE.
```
$ sinfo -n r105u33,r2i4n27
PARTITION  AVAIL   TIMELIMIT NODES  STATE  NODELIST
short      up        4:00:00     1  drain   r2i4n27
short      up        4:00:00     1   down   r105u33
standard   up     2-00:00:00     1  drain   r2i4n27
standard   up     2-00:00:00     1   down   r105u33
long       up     10-00:00:0     1  drain   r2i4n27
long       up     10-00:00:0     1   down   r105u33
bigmem     up     2-00:00:00     1   down   r105u33
gpu        up     2-00:00:00     1   down   r105u33
bigscratch up     2-00:00:00     0    n/a
ddn        up     2-00:00:00     0    n/a
```
### sacct
Use ```sacct``` to view accounting information about jobs AND job steps:
```
$ sacct -j 525198 --format=User,JobID,Jobname,partition,state,time,start,elapsed,nnodes,ncpus
     User        JobID    JobName  Partition      State  Timelimit               Start    Elapsed  NNodes    NCPUS
--------- ------------ ---------- ---------- ---------- ---------- ------------------- ---------- ------- --------
  hpcuser 525198        acct_test      short  COMPLETED   00:01:00 2018-12-19T16:09:34   00:00:54       4      144
          525198.batch      batch             COMPLETED            2018-12-19T16:09:34   00:00:54       1       36
          525198.exte+     extern             COMPLETED            2018-12-19T16:09:34   00:00:54       4      144
          525198.0           bash             COMPLETED            2018-12-19T16:09:38   00:00:00       4        4
```
Use ```sacct -e``` to print a list of fields that can be specified with the ```--format``` option.
### sprio 
By default, ```sprio``` returns information for all pending jobs. Options exist to display specific jobs by JOBID and USER.
```
$ sprio -u hpcuser
  JOBID  PARTITION     USER  PRIORITY   AGE  JOBSIZE PARTITION       QOS
 526752      short  hpcuser  43383470  3733   179737         0  43200000

$ sprio -u hpcuser -n
  JOBID  PARTITION     USER    PRIORITY        AGE    JOBSIZE  PARTITION        QOS
 526752      short  hpcuser  0.01010100  0.0008642  0.0009747  0.0000000  0.1000000
```