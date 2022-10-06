---
layout: default
title: Vermilion
parent: Systems
has_children: true
---

# Vermilion

Vermilion is an OpenHPC-based cluster running on Dual AMD EPYC 7532 Rome CPUs. The nodes run as virtual machines in a local virtual private cloud (OpenStack). Mellanox drivers and OFED are installed on all nodes.

---
## Collaboration / Help
You can get help with Vermilion via email at [HPC-Help@nrel.gov](mailto://hpc-help@nrel.gov) or [live chat](https://teams.microsoft.com/l/channel/19%3a857251ab7f524eb79aa4c44b4579b118%40thread.tacv2/General?groupId=d1c43e0f-8c0f-4de2-80b9-2f57b2ae4203&tenantId=a0f29d7e-28cd-4f54-8442-7885aee7c080).

**Live chat:** <br>
- Users with access to NREL's Teams chat can collaborate via the [Vermilion Users Teams](https://teams.microsoft.com/l/channel/19%3a857251ab7f524eb79aa4c44b4579b118%40thread.tacv2/General?groupId=d1c43e0f-8c0f-4de2-80b9-2f57b2ae4203&tenantId=a0f29d7e-28cd-4f54-8442-7885aee7c080) room.
- **HPC-Help:**<br>
For specific questions about work you're running on Vermillion, send email to hpc-help(@)nrel.gov *and specify **vermilion** on the subject line*.<br>

---
## Connecting to Vermilion
To access vermilion, log into the NREL network and connect via ssh:

    ssh vs.hpc.nrel.gov
    ssh vermilion.hpc.nrel.gov

There are currently two login nodes. They share the same home directory so work done on one will appear on the other. They are:

    vs-login-1
    vs-login-2

You may connect directly to a login node, but they may be cycled in and out of the pool. If a node is unavailable, try connecting to another login node or the `vs.hpc.nrel.gov` round-robin option.

---
## Building code

Don't build or run code on a login node. Login nodes have limited CPU and memory available. Use a compute or GPU node instead. Simply start an interactive job on an appropriately provisioned node and partition for your work and do your builds there. Similarly, build your projects under `/projects/your_project_name/` as home directories are **limited to 5GB** per user.


---

