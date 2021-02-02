---
title: December 2020 NREL HPC Monthly Update
date: 2020-12-04
layout: default
brief: FY20 Expired Projects' Data, Changes to MSS, ESIF-HPC-3 Project Update
---

# FY20 Expired Projects' Data
Reminder that FY20 expired Projects' data will be removed from Eagle on January 1st, 2021.  Any data needed needs to either be copied to the new AWS MSS or other arrangements made outside of HPC.

Due to vendor ending support for the old MSS equipment, the new HPC Mass Storage System (MSS) environment will reside on Amazon Web Services. The old Gyrfalcon MSS data will be made read-only on December 1st, 2020. Any data 15 months old or less will be migrated to AWS MSS. 
 

# Changes to Eagle Mass Storage System (MSS)
*What:* NREL HPC Operations is in the process of retiring the on-premise MSS capability and has started using cloud-based data storage capability.

*Why:* The vendor has announced end-of-life for the technology previously used to provide MSS.  In exploring alternatives, a cloud-based solution leverages expertise of the Advanced Computing Operations team and is significantly more cost effective. Adequate bandwith is available to/from Cloud (10x more bandwith than to/from the previous on-premise MSS).  Also a very small percentage of data written to MSS is ever read again, thus prompting a change to a fixed 15-month retention from when data is written to MSS.

*When/How:* On December 1st, all new writes to MSS will be to cloud-based storage.  Reading data from the existing on-premise MSS capability will be supported through March 31, 2021.  Active data (written or last read within the last 15 months prior to December 1st, 2020) will be migrated from the on-premise MSS to the new cloud-based storage.

*What stays the same:* MSS is an additional location available to projects active on Eagle to keep and protect important data in addition to the Eagle high-performance storage (/projects, /shared-projects, /datasets)

*What changes:* Data will be retained for 15 months from the date written.  This differs from the current retention policy of minimum 15 months with deletion if needed. Restore requests of MSS data that are cloud-based will initially require a request to the HPC Help Desk, and may require 48 hours to be able to recover.
 

# ESIF-HPC-3 Project Update

The ESIF-HPC-3 project moves on. The Request for Proposals and its many pieces (including the technical specifications, benchmark suite and specifications, and workload analysis) should be live by the time you read this. At this time, our hope is that vendors are busy designing the next-generation computing and storage systems that will serve EERE research starting in FY23, and preparing proposals for review by a cross-Directorate NREL Source Evaluation Team starting in mid-January 2021.

