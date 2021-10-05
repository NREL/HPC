---
title: November 2020 Monthly Update
date: 2020-11-03
layout: default
brief: Lustre quotas, changes to MSS, and ESIF-HPC-3
---

# ESIF-HPC-3 Project Update
The ESIF-HPC-3 project has begun! The effort to acquire Eagle’s successor involves ongoing engagement with stakeholders (EERE, Lab Program Management, and the HPC user community), tracking industry trends, analysis of Eagle’s workload over its life to date, external design review, and a carefully managed Request for Proposals targeted for release later this year. We are currently in the process of completing the draft technical specifications for the ESIF-HPC-3 system, and are targeting the start of FY23 for general production access.

If you would like to weigh in on how your work would benefit from existing or new features you could envision, please feel free to send a note to hpc-help@nrel.gov. We will open a discussion on the draft design in its current form if there is sufficient interest.
 

# Lustre Quotas
Effective with the new Fiscal Year 2021 Project allocations for Eagle, quotas for approved storage allocations' capacities have been implemented on /projects and MSS on Eagle. This was to encourage users to manage their /projects data usage and usage of /scratch for jobs. HPC Operations is developing reporting capabilities of usage, but in the mean time, users may request help from the HPC Help Desk, or utilize these procedures from an Eagle Login node:

To view project quotas and usage:

Get the ProjectID for your /projects directory:
lfs project -d /projects/csc000
 
110255 P /projects/csc000
 
Get the usage and quota in kbytes:
lfs quota -p 110255 /projects/csc000
 
Disk quotas for prj 110255 (pid 110255):
 
       Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
 
/projects/csc000 3165308*   3072    4096       -      48  1073741824 2147483648       -
An * means you have exceeded your soft quota of 3072kb, the hard limit of 4096kb reached means no more writes are allowed. Grace period is set to default of 7 days but will show time until writes are suspended. "files" indicate the number of inodes used and soft and hard limits.

We encourage users to run their jobs in Eagle /scratch and copy results and other necessary project files to /projects, possibly using tar and zip to conserve space (tar czvf tar-file-name.tgz source-directory-files-to-tar-and zip).

If you are over your project quota, we recommend removing unneeded files and directories or moving them to your /scratch directory until no longer needed.  Remember /scratch files are purged regularly per NREL’s HPC storage retention policies.
 

# Changes to Eagle Mass Storage System (MSS)
What: NREL HPC Operations is in the process of retiring the on-premise MSS capability and has started using cloud-based data storage capability.

Why: The vendor has announced end-of-life for the technology previously used to provide MSS.  In exploring alternatives, a cloud-based solution leverages expertise of the Advanced Computing Operations team and is significantly more cost effective. Adequate bandwith is available to/from Cloud (10x more bandwith than to/from the previous on-premise MSS).  Also a very small percentage of data written to MSS is ever read again, thus prompting a change to a fixed 15-month retention from when data is written to MSS.

When/How: On December 1st, all new writes to MSS will be to cloud-based storage.  Reading data from the existing on-premise MSS capability will be supported through March 31, 2021.  Active data (written or last read within the last 15 months prior to December 1st, 2020) will be migrated from the on-premise MSS to the new cloud-based storage.

What stays the same: MSS is an additional location available to projects active on Eagle to keep and protect important data in addition to the Eagle high-performance storage (/projects, /shared-projects, /datasets)

What changes: Data will be retained for 15 months from the date written.  This differs from the current retention policy of minimum 15 months with deletion if needed. Restore requests of MSS data that are cloud-based will initially require a request to the HPC Help Desk, and may require 48 hours to be able to recover.

Also a reminder:  Project data (/projects) for FY20 projects not continuing into FY21 will have until December 31, 2020 to move data off Eagle to MSS or other long-term storage, before it is purged from Eagle on January 1st, 2021.
 
