---
title: January 2021 Monthly Update
data: 2021-01-04
layout: default
brief: Arbiter2, ESIF-HPC-3, and application updates
---

# Arbiter2 

On Tuesday, January 12, we will be upgrading the Arbiter2 software on the Eagle login nodes. The upgrade improves stability of the program, as well as fixes some broken features.

Arbiter2 limits individual resources on these shared resources within a range. Certain processes (for example, those related to code compilation) are exempt from these limits. For other processes, consistently high processor utilization leads to resource throttling in order to equalize the net amount of resource users have access to over time. As usage returns below a level consistent with the smooth operation of the shared login node, the throttling is relaxed. Users exceeding per-user resource limits on login nodes ("in violation") will receive emails when they trigger a violation, and when their usage returns below limits. From users' perspective the upgrade will not change limits or throttling behavior, it will just turn on notifications.

# ESIF-HPC-3 Project Update

The ESIF-HPC-3 Request for Proposals is live! For those interested, the content can be found on SAM.gov.

# Application Updates

* Q-Chem has been upgraded to version 5.3.2. See changes here.
* Star-CCM version 15.06.008 is available on Eagle.
* ARM Forge version 20.2 is available on Eagle.

We are working on acquiring a Maintenance license for VASP 6. Once we have this in place, users will need to have an upgraded VASP 6 Research workgroup license in order to use our VASP 6 builds on Eagle.

