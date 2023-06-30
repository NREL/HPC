---
layout: default
title: File Systems 
has_children: true
hide:
 - toc
---

# About the Eagle Cluster

Eagle is configured to run compute-intensive and parallel computing jobs. It is a cluster comprised of 2604 nodes (servers) that run the Linux operating system (Red Had Linux or the derivative CentOS distribution), with a peak performance of 8 PetaFLOPS.

Please see the [System Configurations](../index.md) page for more information about hardware, storage, and networking.

## Accessing Eagle
Access to Eagle requires an NREL HPC account and permission to join an existing allocation. Please see the [System Access](https://www.nrel.gov/hpc/system-access.html) page for more information on accounts and allocations.

#### For NREL Employees:

Users on an NREL device may connect via ssh to **eagle.hpc.nrel.gov** from the NREL network. This will connect to one of the three login nodes. Users also have the option of connecting directly to an individual login node using one of the following names: 

* el1.hpc.nrel.gov
* el2.hpc.nrel.gov
* el3.hpc.nrel.gov

#### For External Collaborators:
If you are an external HPC user, you will need a [One-Time Password Multifactor token (OTP)](https://www.nrel.gov/hpc/multifactor-tokens.html) for two-factor authentication.

For command line access, you may login directly to **eagle.nrel.gov**. Alternatively, you can connect to the [SSH gateway host](https://www.nrel.gov/hpc/ssh-gateway-connection.html).  If you need to use web-based applications, X11 applications, or perform file transfers on non-Eagle systems, connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html). 

## Get Help With Eagle
Please see the [Help and Support Page](../../help.md) for further information on how to seek assistance with Eagle or your NREL HPC account. 