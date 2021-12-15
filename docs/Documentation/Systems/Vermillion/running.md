---
 layout: default
 title: Running on Vermillion
 parent: Vermillion
 grand_parent: Systems
---

# Running on Vermillion

Please see the [Modules](./modules.md) page for information about setting up your environment and loading modules. 


## Login nodes

```
vs.hpc.nrel.gov 
vermilion.hpc.nrel.gov
vs-login-1.hpc.nrel.gov
vs-login-2.hpc.nrel.gov
```


## Slurm and Partitions

A list of partitions can be returned by sunning the `sinfo` command. Partitions are distinguished by the number of virtual cores per node.

| Partition Name | Cores per node |
|----------------|----------------|
|       t        |       4        |
|       sm       |       16       |
|      std       |       30       |
|     large      |       60       |

In addition to these CPU-only options, partition "gpu" has nodes with 30 cores and 1 nVidia A100 GPU.
