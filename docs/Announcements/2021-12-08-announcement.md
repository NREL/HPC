---
title: December 2021 Monthly Update
data: 2021-12-08
layout: default
brief: Data Security, System Time, Data Storage Policy
---

# Data Security Policy Reminder
Eagle and systems in the ESIF data center are managed under a "low" Authority to Operate (ATO) per the [FIPS 199](https://csrc.nist.gov/csrc/media/publications/fips/199/final/documents/fips-pub-199-final.pdf) standard. End users should be familiar with NREL's HPC [Data Security Policy](https://www.nrel.gov/hpc/data-security-policy.html) for this class of systems. The potential impact rating of data is the responsibility of the data owner.

The most common data security risk on Eagle is misconfiguration or misunderstanding of file permissions. This may involve accidentally setting UNIX ownership and/or permissions or ACLs on a directory that make files readable outside of the intended audience, or failing to remove permissions from users that should no longer have access to files.

HPC Leads and PI’s can mitigate any potential data exposure or leaks by checking file and directory ownerships and permissions, as well as updating their project entitlement in Lex as needed. Alternatively, HPC Leads or PI's could consider utilizing the AWS cloud environment "[Stratus](https://thesource.nrel.gov/cloud-computing/)" as it has been classified for "moderate impact" data.

File and folder permissions are an important way to you projects data from unintended access outside the project. Due to the number of people with privileged access to on ESIF data center systems, they are not a reasonable control for data rated outside of low. Data that is covered under a CRADA must be agreed upon by the legal entities that signed the CRADA as to what are the appropriate controls for that data.

We understand that data classification can be challenging, and security requirements vary from project to project. We recommend project PI's contact the NREL Legal Department or their respective legal department with any questions regarding the classification of their data and where it may be safely stored.

# Eagle Second Quarter System Time
The Eagle cluster will be offline for regular scheduled maintenance for the week beginning, January 10th at approximately 6:00am (Mountain), and will return to service on January 13th. Eagle login nodes, DAV/FastX nodes, all filesystems (including lustre and /home), Globus, and related support systems will be unavailable during this time.

Network maintenance is planned during this time as well, and access to certain internet/external-facing HPC services (including eagle.nrel.gov, eagle-dav.nrel.gov, and the [self-service password tool](https://hpcusers.nrel.gov/) will be temporarily unavailable and all outbound traffic will have no access to the internet from the HPC datacenter. 

# Reminder of Eagle Data Storage Policies for FY21 projects that ended on 9/30/2021
Eagle usage policies can be found on the [Policies Page](https://www.nrel.gov/hpc/policies.html)

Users are always strongly encouraged to remove any data on Eaglefs that is not needed, to benefit other users of this shared resource. Eaglefs consists of /shared-projects, /datasets, /scratch and /projects.

In summary, data in /projects for allocations that end on 9/30/2021 will be purged after 12/31/2021.

Users may continue to log in to HPC systems for a period of 3 months after the project enters the Expired state to move relevant data off of HPC primary storage (primarily /projects/<project_handle>) to another storage location. The ability to write new data to /projects has been disabled for those projects that have expired. Instructions on how to archive data using AWS MSS (Mass Storage System) can be found here.

Users may continue to request MSS files that have been archived, for a period of 15 months after the files have been initially archived.

Eagle's /scratch files have a policy of potentially being purged if not accessed within 28 days.


