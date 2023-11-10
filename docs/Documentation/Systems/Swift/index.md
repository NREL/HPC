---
layout: default
title: Swift
parent: Systems
has_children: true
hide:
 - toc
---

# About the Swift Cluster

Swift is an AMD-based HPC cluster with AMD EPYC 7532 (Rome) CPU's that supports EERE Vehicles Technologies Office (VTO) projects. Any VTO funded EERE project is eligible for an allocation on Swift. Allocation decisions are made by EERE through the annual allocation cycle. Swift is well suited for parallel jobs up to 64 nodes and offers better queue wait times for projects that are eligible.

Please see the [System Configurations](https://nrel.github.io/HPC/Documentation/Systems/) page for more information about hardware, storage, and networking.

## Accessing Swift
Access to Swift requires an NREL HPC account and permission to join an existing allocation. Please see the [System Access](https://www.nrel.gov/hpc/system-access.html) page for more information on accounts and allocations.

#### Login Nodes:
```
swift.hpc.nrel.gov
swift-login-1.hpc.nrel.gov
```
#### For NREL Employees:
Swift can be reached from the NREL VPN via ssh to the login nodes as above.

#### For External Collaborators:
There are currently no external-facing login nodes for Swift. There are two options to connect:

1. Connect to the [SSH gateway host](https://www.nrel.gov/hpc/ssh-gateway-connection.html) and log in with your username, password, and OTP code. Once connected, ssh to the login nodes as above.
1. Connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) and ssh to the login nodes as above.

## Get Help With Swift
Please see the [Help and Support Page](../../help.md) for further information on how to seek assistance with Swift or your NREL HPC account. 


