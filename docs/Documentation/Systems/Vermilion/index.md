---
layout: default
title: Vermilion
parent: Systems
has_children: true
---

# About Vermilion

Vermilion is an OpenHPC-based cluster running on Dual AMD EPYC 7532 Rome CPUs and nVidia A100 GPUs. The nodes run as virtual machines in a local virtual private cloud (OpenStack). Vermilion is allocated for NREL workloads and intended for LDRD, SPP or Office of Science workloads. Allocation decisions are made by the IACAC through the annual allocation request process. Check back regularly as the configuration and capabilities for Vermilion are augmented over time.

## Accessing Vermilion
Access to Vermilion requires an NREL HPC account and permission to join an existing allocation. Please see the [System Access](https://www.nrel.gov/hpc/system-access.html) page for more information on accounts and allocations.

#### For NREL Employees:
To access vermilion, log into the NREL network and connect via ssh:

    ssh vs.hpc.nrel.gov
    ssh vermilion.hpc.nrel.gov

#### For External Collaborators:
There are currently no external-facing login nodes for Vermilion. There are two options to connect:

1. Connect to the [SSH gateway host](https://www.nrel.gov/hpc/ssh-gateway-connection.html) and log in with your username, password, and OTP code. Once connected, ssh to the login nodes as above.
1. Connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) and ssh to the login nodes as above.

There are currently two login nodes. They share the same home directory so work done on one will appear on the other. They are:

    vs-login-1
    vs-login-2

You may connect directly to a login node, but they may be cycled in and out of the pool. If a node is unavailable, try connecting to another login node or the `vs.hpc.nrel.gov` round-robin option.

## Get Help with Vermilion 

Please see the [Help and Support Page](../../help.md) for further information on how to seek assistance with Vermilion or your NREL HPC account. 

## Building code

Don't build or run code on a login node. Login nodes have limited CPU and memory available. Use a compute or GPU node instead. Simply start an interactive job on an appropriately provisioned node and partition for your work and do your builds there. Similarly, build your projects under `/projects/your_project_name/` as home directories are **limited to 5GB** per user.


---

