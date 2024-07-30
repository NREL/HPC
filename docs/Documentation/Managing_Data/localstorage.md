# Local and Scratch Storage on NREL HPC Systems

The table below summarizes the local and scratch storage currently on NREL HPC systems. 

| System Name | Node Local Storage | $TMPDIR Default | Default $TMPDIR Storage Type | Global Scratch Storage |
| -- | -- | -- | -- | -- | 
| Kestrel | 1.7TB on 256 of the standard compute nodes, 5.8TB on bigmem nodes, 3.2TB GPU nodes. Other nodes have none. | `/tmp/scratch/$SLURM_JOBID` | Local disk when available, or RAM | `/scratch/$USER` (Lustre) | 
| Swift | 1.8TB | `/scratch/$USER/$SLURM_JOBID` | Local disk | None | 
| Vermilion | 60GB (t), 250GB (sm), 500GB (std), 1.0TB (lg), 2.0TB (gpu) | `/tmp` | RAM. Write to `/tmp/scratch` instead to use local disk. | `/scratch/$USER` |


**Important Notes**

- Local storage is local to a node and usually faster to access by the processes running on the node. Some scenarios in which using the local disk might make your job run faster are:
    - Your job may access or create many small (temporary) files
    - Your job may have many parallel tasks accessing the same file
    - Your job may do many random reads/writes or memory mapping.
- Local or scratch spaces are for temporary files only and **there is no expectation of data longevity in these spaces. HPC users should copy results from those spaces to a `/projects` or global scratch directory as part of the job script before the job finishes**.
- A node will not have read or write access to any other node's local scratch, only its own
- On Kestrel, the path `/tmp/scratch` is not writeable. Use `$TMPDIR` instead.
- On Kestrel, only 256 of the standard compute nodes have real local disk, the other standard compute nodes have **no local disk space**. For the nodes without local storage, writing to `$TMPDIR` uses RAM. This could **cause an out-of-memory error if using a lot of space in $TMPDIR**. To solve this problem:
    - Use `/scratch/$USER` instead of the default `$TMPDIR` path if the job benefits little from local storage (e.g. jobs with low I/O communication)
    - Request nodes with local storage by using the `--tmp` option in your job submission script. (e.g. `--tmp=1600000`). Then, `$TMPDIR` will be using a local disk. 
    - In addition, on Kestrel, this bash command can be used to check if there is a local disk on the node: "`if [ -e /dev/nvme0n1 ]`". **This will only work on standard compute nodes**. For example:

```    
if [ -e /dev/nvme0n1 ]; then
 echo "This node has a local storage and will use as the scratch path"
 APP_SCRATCH=$TMPDIR
else
 echo "This node does not have a local storage drive and will use /scratch as the scratch path"
 APP_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
fi
```
*This does not work on bigmem nodes. All bigmem nodes have a real local disk.*

