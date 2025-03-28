---
layout: default
title: Systems
has_children: true
order: 4
---

# NREL Systems
NREL operates three on-premises systems for computational work. 

## System Configurations

| Name        | Kestrel |  Swift        | Vermilion     | 
| :---------- | :------ |  :----------- | :------------- |
| OS          | RedHat Enterprise Linux |  Rocky Linux    | RedHat       |
| Login       | kestrel.hpc.nrel.gov |  swift.hpc.nrel.gov | vs.hpc.nrel.gov |
| CPU         | Dual socket Intel Xeon Sapphire Rapids |  Dual AMD EPYC 7532 Rome CPU | Dual AMD EPYC 7532 Rome CPU |
| Cores per CPU Node | 104 cores |  128 cores | Varies by partition | 
| Interconnect | HPE Slingshot 11 | InfiniBand HDR| 25GbE |
| HPC scheduler | Slurm | Slurm | Slurm |
| Network Storage | 95PB Lustre | 3PB NFS | 440 TB
| GPU         | 156 4x NVIDIA H100 SXM GPUs | 10 4x NVIDIA A100 40GB GPUs | 5 nodes Single A100
| Memory      | 256GB, 384GB, 700GB, 2TB | 256GB(CPU) 1T(GPU) | 256GB (base)
| Number of Nodes| 2478 | 484 | 133 virtual |


