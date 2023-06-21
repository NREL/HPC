# Eagle Job Partitions and Scheduling Policies
*Learn about job partitions and policies for scheduling jobs on Eagle.*

## Partitions

Eagle nodes are associated with one or more partitions.  Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, per-node local scratch disk requirements, and whether graphics processing units (GPUs) are needed.

Jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**

The following table summarizes the partitions on Eagle.

| Partition Name | <div style="width:1px">Description</div>   | Limits | Placement Condition |
| -------------- | ----------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes <br> with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is 4 GPU nodes 2 Bigmem nodes 7 standard nodes 13 total nodes | 1 job with a max of 2 nodes per user 01:00:00 max walltime | ```-p debug``` <br>   or<br>   ```--partition=debug``` |


