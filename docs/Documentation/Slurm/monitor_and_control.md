# Commands to Monitor and Control Jobs

Slurm includes a suite of command-line tools used to submit, monitor, and control jobs and the job queue.


| Command | Description |
| ----------| ------------ | 
| ```squeue``` | Show the Slurm queue. Users can specify JOBID or USER.|
| ```scontrol``` | 	Controls various aspects of jobs such as job suspension, re-queuing or resuming jobs and can display diagnostic info about each job.|
| ```scancel``` | Cancel specified job(s). |
| ```sinfo``` | View information about all Slurm nodes and partitions. |
| ```sacct``` | Detailed information on accounting for all jobs and job steps. |
| ```sprio``` | View priority and the factors that determine scheduling priority. |

Please see ```man``` pages on the cluster for more information on each command. Also see ```--help``` or ```--usage``` flags for each.

Our [Presentation on Advanced Slurm Features](https://www.nrel.gov/hpc/assets/pdfs/slurm-advanced-topics.pdf) is also available as a resource, which has supplementary information on how to manage jobs.

Another great resource for Slurm at NREL is [this repository on Github](https://github.com/sayerhs/nrel-eagle/blob/master/nrel-eagle.md).

## Usage Examples

### squeue

The `squeue` command is used to view the current state of jobs in the queue. 

To show your jobs:

```
$ squeue -u hpcuser
           JOBID    PARTITION       NAME      USER   ST       TIME      NODES   NODELIST(REASON)
          506955          gpu   wait_tes   hpcuser   PD       0:00          1      (Resources)
```

To show all jobs in the queue with extended information:

```
$ squeue -l
Thu Dec 13 12:17:31 2018
 JOBID  PARTITION NAME     USER     STATE   TIME    TIME_LIMIT   NODES  NODELIST(REASON)
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

Note that the Slurm start times are only an estimate, and are updated frequently based on the current state of the queue and the specified `--time` of all jobs in the queue. 

```

$ squeue --start -j 509851,509852
 JOBID    PARTITION    NAME      USER      ST          START_TIME    NODES   SCHEDNODES   NODELIST(REASON)
 509851   short      test1.sh   hpcuser    PD                 N/A      100       (null)       (Dependency)
 509852   short      test2.sh   hpcuser    PD 2018-12-19T16:54:00        1      r1i6n35         (Priority)
 
```

#### Output Customization of the squeue Command

The displayed fields in `squeue` can be highly customized to display the information that's most relevant for the user by using the `-o` or `-O` flags. The full list of customizable fields can be found under the entries for these flags in the `man squeue` command on the system. 

By setting the environment variable export $SQUEUE_FORMAT, you can override the system's default squeue fields with your own. For example, if you run the following line (or place it in your `~/.bashrc` or `~/.bash_aliases` file to make it persistent across logins):

`export SQUEUE_FORMAT="%.18i %.15P %.8q %.12a %.8p %.8j %.8u %.2t %.10M %.6D %R"`

Using `squeue` will now provide the formatted output:

```
JOBID    PARTITION   QOS    ACCOUNT   PRIORITY     NAME     USER    ST     TIME    NODES NODELIST(REASON)
13141110 standard   normal  csc000    0.051768    my_job   hpcuser  R   2-04:01:17   1    r1i3n29
```

Or you may wish to add the `%V` to show the timestamp that a job was submitted, and sort by timestamp, ascending:

`squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %20V %6q %12l %R" -S "V"`

Example output:

```
             JOBID PARTITION     NAME     USER ST       TIME  NODES SUBMIT_TIME          QOS    TIME_LIMIT   NODELIST(REASON)
          13166762    bigmem    first  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13166761    bigmem       P5  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13166760    bigmem       P4  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13166759    bigmem      Qm3  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13166758    bigmem       P2  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13166757    bigmem       G1  hpcuser PD       0:00      1 2023-08-30T14:08:11  high   2-00:00:00   (Priority)
          13167383    bigmem       r8  hpcuser PD       0:00      1 2023-08-30T16:25:52  high   2-00:00:00   (Priority)
          13167390  standard      P12  hpcuser PD       0:00      1 2023-08-30T16:25:55  high   2-00:00:00   (Priority)
          13167391    bigmem      P34  hpcuser PD       0:00      1 2023-08-30T16:25:55  high   2-00:00:00   (Priority)
          13167392    bigmem    qchem  hpcuser PD       0:00      1 2023-08-30T16:25:55  high   2-00:00:00   (Priority)
          13167393     debug  testrun  hpcuser PD       0:00      1 2023-08-30T16:25:55  high   2-00:00:00   (Priority)
          13167394    bigmem   latest  hpcuser PD       0:00      1 2023-08-30T16:25:55  high   2-00:00:00   (Priority)
          13182480     debug  runtest  jwright2 R      31:01      1 2023-09-01T14:49:54  normal 59:00        r3i7n35
```

Many other options are available in the man page.


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

Use the `-n` flag to provide a normalized priority weighting with a value between 0-1:

$ sprio -u hpcuser -n
  JOBID  PARTITION     USER    PRIORITY        AGE    JOBSIZE  PARTITION        QOS
 526752      short  hpcuser  0.01010100  0.0008642  0.0009747  0.0000000  0.1000000
```

The sprio command also has some options that can be used to view the entire queue by priority order. The following command will show the "long" (`-l`) format sprio will extended information, sorted by priority in descending order (`-S -Y`), and piped through the `less` command with line numbers shown on the far left (`less -N`):

`sprio -S -Y -l | less -N`

```
1           JOBID PARTITION     USER   PRIORITY       SITE        AGE      ASSOC  FAIRSHARE    JOBSIZE  PARTITION        QOS        NICE                 TRES
2        13150512 standard-  hpcuser  373290120          0    8909585          0  360472143      84743    3823650          0           0
3        13150514 standard-  hpcuser  373290070          0    8909534          0  360472143      84743    3823650          0           0
```

When `sprio` is piped through the `less` command for paginating, press the `/` key and type in a jobid or a username and press the return key to search for and jump to that jobid or username. Press `/` and hit return again to search for the next occurrence of your search term, or use the `?` instead of `/` to search upwards in the list. Press q to exit.

Note that when piped through `less -N`, line numbers may be equated to position in the priority queue plus 1, because the top column label line of `sprio` is counted by `less`. To remove the column labels from `sprio` output, add the `-h` or `--noheader` flag to `sprio`.

The `-l`(`--long`) flag precludes using the `-n` for normalized priority values. 

Like `squeue` and other Slurm commands, `sprio` supports the `-o` format flag to customize the columns that are displayed. For example:

`sprio S -Y -o "%i %r %u %y"`

Will show only the jobid, partition, username, and normalized priority. More details about output formatting are available in `man sprio`.


