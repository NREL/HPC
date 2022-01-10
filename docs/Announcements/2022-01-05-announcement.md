---
title: January 2022 Monthly Update
data: 2022-01-05
layout: default
brief: System Time, Data Storage Policy, Survey Thanks, Comsol Event
---
# January 2022 System Time Reminder
Eagle will be offline for regularly scheduled maintenance starting on Monday, January 10th, 2022 at 6:00am (Mountain Time.) The system is expected to return to service by January 13th, 2022.

During this outage, the Eagle login nodes, DAV/FastX nodes, all filesystems (including lustre and /home), Globus, and other related support systems will be unavailable.  Notable tasks for this system time include an upgrade to the Slurm scheduler, filesystem maintenance, cooling systems maintenance and repair, updates to FastX on ed7, and adjustments to the Arbiter2 resource monitor.

Network maintenance is planned during this time as well. Access to certain internet/external-facing HPC services (including eagle.nrel.gov, eagle-dav.nrel.gov, and the self-service password tool at https://hpcusers.nrel.gov/) will be temporarily unavailable, and all outbound traffic will have no access to the internet from the HPC data center.

# Final Reminder of Eagle Data Storage Policies for FY21 projects that ended on 9/30/2021
Eagle usage policies can be found on the [HPC website policy page](https://www.nrel.gov/hpc/policies.html).

In summary, data in /projects for allocations that end on 9/30/2021 will be purged after 12/31/2021.  Users may continue to log in to HPC systems for a period of 3 months after the project enters the Expired state to move relevant data off of HPC primary storage (primarily /projects/<project_handle>) to another storage location. The ability to write new data to /projects has been disabled for those projects that have expired. Instructions on how to archive data using AWS MSS (Mass Storage System) can be found on the Mass Storage page.  Users may continue to request MSS files that have been archived, for a period of 15 months after the files have been initially archived.  Eagle's /scratch files have a policy of potentially being purged if not accessed within 28 days.

Users are always strongly encouraged to remove any data on Eaglefs that is not needed, to benefit other users of this shared resource. Eaglefs consists of /shared-projects, /datasets, /scratch and /projects.

# Thank you for participating in the annual survey
We are extremely grateful to you for contributing your valuable time, your honest feedback, and your thoughtful suggestions.  We are committed to utilizing the information to implement worthwhile improvements to the environment and our processes to make the cloud experience more efficient.  We will share these implementations with you through our monthly newsletter.  

# Applications
COMSOL will provide NREL an overview of the new COMSOL 6.0 release on January 25, 2022, from 11:00 AM - 12:00 PM MST.

To register, go to the [COMSOL event registration page](https://www.comsol.com/events/web-meeting/introduction-to-comsol-multiphysics-version-60-for-nrel-99332).

Attend to learn about the latest features, and ask questions about COMSOL and NREL-HPC.
