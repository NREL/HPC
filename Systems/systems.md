---
layout: default
title: Systems
has_children: true
order: 4
---

# NREL Systems
NREL operates three on-premises systems for computational work.

## System configurations

| Name        | Eagle     | Swift          | Vermillion     |
| :---------- | :-------- | :-----------   | :------------- |
| OS          | CentOS    | Rocky Linux    | CentOS 8     |
| CPU         | Dual Intel Xeon Gold Skylake 6154 | Dual AMD EPYC 7532 Rome CPU | Dual AMD EPYC 7532 Rome CPU |
| Interconnect | InfiniBand EDR | InfiniBand HDR| 2x25GbE RDMA |
| HPC scheduler | Slurm | Slurm | Slurm |
| Network Storage | 17PB Lustre FS | 3PB NFS | 440 TB |
| GPU | Dual NVIDIA Tesla V100 | None | 8 nodes Single A100 |
| Memory | 96GB, 192GB, 768GB | 256GB | <pre>Part  Mem<br> gpu  114 GB <br>  lg  229 GB <br> std  114 GB <br>  sm   61 GB <br>   t   15 GB
| Number of Nodes | 2618 | 484 | 133 |

