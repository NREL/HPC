---
layout: default
title: Systems
has_children: true
order: 4
---

# NREL Systems
NREL operates three on-premises systems for computational work. 

## System configurations

| Name        | Eagle     | Swift        | Vermillion     |
| :---------- | :-------- | :----------- | :------------- |
| OS          | CentOS    | Rocky Linux    | RedHat       |
| Login       | eagle.hpc.nrel.gov | swift.hpc.nrel.gov | vs.hpc.nrel.gov |
| CPU         | Dual Intel Xeon Gold Skylake 6154 | Dual AMD EPYC 7532 Rome CPU | Dual AMD EPYC 7532 Rome CPU |
| Interconnect | InfiniBand EDR | InfiniBand HDR| 25GbE |
| HPC scheduler | Slurm | Slurm | Slurm |
| Network Storage | 17PB Lustre FS | 3PB NFS |
| GPU | Dual NVIDIA Tesla V100 | None | 8 nodes Single A100
| Memory | 96GB, 192GB, 768GB | 256GB | vs-gpu-000[01-06] 114 GB <br> vs-lg-00[01-18] 229 GB <br> vs-sm-00[01-32] 61 GB <br> vs-std-00[01-62] 114 GB <br> vs-t-00[01-15] 15 GB
| Number of Nodes | 2618 | 484 | 133 virtual |
