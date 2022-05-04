---
title: May 2022 Monthly Update
data: 2022-05-04
layout: default
brief: FY23 Allocations, Intro to HPC Workshops, Windows SSH, Lustre Usage, Documentation
---

# FY23 HPC Allocation Process
The Eagle allocation process for FY23 is scheduled to open up on May 11, with applications due June 8. The application process will be an update of the process used in FY22, with additional information requested to help manage the transition from Eagle to Kestrel. Be sure to sign-up for the webinar on May 17 from 11-12 a.m. MT. Michael Martin, Staff Scientist in the Computational Science Center at NREL, will be presenting on the allocation process, key dates, changes from last year, and how to submit a request.  

The [registration site](https://nrel.zoomgov.com/meeting/register/vJItf-igpjouG2cx1ng4tinceiPhBw5Ufz4) for the webinar is now available.

# NREL HPC Workshops - Intro to HPC Series (Save the Dates)

NREL HPC Operations and Application Support teams will host an Intro to HPC workshop series this June every Wednesday from 12-1 p.m.  Webinar information to follow.

Topics/Schedule:

* Linux Fundamentals: Utilizing the Command Line Interface on June 1st 12-1
* NREL HPC Systems on June 8th 12-1
* Resource Management: Slurm on June 15th 12-1
* Software Environments on June 22nd 12-1
* JupyterHub on June 29th 12-1


# Workaround for Windows SSH Users
Some people who use Windows 10/11 computers to ssh to Eagle from a Windows command prompt, powershell, or via Visual Studio Code's SSH extension have received a new error message about a "Corrupted MAC on input" or "message authentication code incorrect." This error is due to an outdated OpenSSL library included in Windows and a recent security-mandated change to ssh on Eagle. However, there is a functional workaround for this issue. (Note: If you are not experiencing the above error, you do not need and should not use the following workaround.)

For command-line and Powershell ssh users, adding "-m hmac-sha2-512" to your ssh command will resolve the issue. For example: "ssh -m hmac-sha2-512 <username>@eagle.hpc.nrel.gov"

For VS Code SSH extension users, you will need to create an ssh config file on your local computer (~/.ssh/config), with a host entry for Eagle that specifies a new message authentication code: 
```
Host eagle
    HostName eagle.hpc.nrel.gov
    MACs hmac-sha2-512
```

The configuration file will also apply to command-line ssh in Windows, as well. This [Visual Studio Blog post](https://code.visualstudio.com/blogs/2019/10/03/remote-ssh-tips-and-tricks) has further instructions on how to create the ssh configuration file for Windows and VS Code.

# Lustre Filesystem Usage Reminder
The Lustre file systems that hosts /projects, /scratch, /shared-projects and /datasets works most efficiently when it is under 80% full. Please do your part to keep the file system under 80% by cleaning up your /projects, /scratch and /shared-projects spaces.

# Documentation
We would like to announce our user-contributed [documentation repository](https://github.com/NREL/HPC) and [website](https://nrel.github.io/HPC/) for Eagle and other NREL HPC systems that is open to both NREL and non-NREL users. This repository serves as a collection of code examples, executables, and utilities to benefit the NREL HPC community. It also hosts a site that provides more verbose documentation and examples.  If you would like to contribute or recommend a topic to be covered please open an issue or a pull request in the repository. Our [contribution guidelines](https://github.com/NREL/HPC/blob/master/CONTRIBUTING.md) offer more detailed instructions on how to add content to the pages.
