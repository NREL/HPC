---
layout: default
title: Systems
has_children: true
order: 4
---

# NREL Systems
NREL operates three on-premises systems for computational work. 

## System configurations

| Name        | Kestrel | Eagle     | Swift        | Vermilion     | 
| :---------- | :------ | :-------- | :----------- | :------------- |
| OS          | RedHat Enterprise Linux | CentOS    | Rocky Linux    | RedHat       |
| Login       | kestrel.hpc.nrel.gov | eagle.hpc.nrel.gov | swift.hpc.nrel.gov | vs.hpc.nrel.gov |
| CPU         | Dual socket Intel Xeon Sapphire Rapids | Dual Intel Xeon Gold Skylake 6154 | Dual AMD EPYC 7532 Rome CPU | Dual AMD EPYC 7532 Rome CPU |
| Interconnect | HPE Slingshot 11 | InfiniBand EDR | InfiniBand HDR| 25GbE |
| HPC scheduler | Slurm | Slurm | Slurm | Slurm |
| Network Storage | 95PB Lustre | 17PB Lustre FS | 3PB NFS | 440 TB
| GPU         | 132 4x NVIDIA H100 SXM GPUs | Dual NVIDIA Tesla V100 | None | 5 nodes Single A100
| Memory      | 256GB, 384GB, 2TB | 96GB, 192GB, 768GB | 256GB | 256GB (base)
| Number of Nodes| 2454 | 2618 | 484 | 133 virtual |



