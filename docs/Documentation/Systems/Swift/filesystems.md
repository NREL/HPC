---
layout: default
title: Filesystems
parent: Systems
has_children: true
---

# Swift Filesystem Architecture Overview

Swift's central storage currently has a capacity of approximately 3PB, served over NFS (Network File System). It is a performant system with 
multiple read and write cache layers and redundancies for data protection, but it is not a parallel filesystem, unlike Eagle's Lustre configuration.

The underlying filesystem and volume management is via ZFS. Data is protected in ZFS RAID arrangements (raidz3) of 8 storage disks and 3 parity disks. 

Each Swift fileserver serves a single storage chassis (JBOD, "just a bunch of disks") consisting of multiple spinning disks plus SSD drives for read and write caches. 

Each fileserver is also connected to a second storage chassis to serve as a redundant backup in case the primary fileserver for that storage chassis fails, allowing continued access to the data on the storage chassis until the primary fileserver for that chassis is restored to service.

## Project Storage: /projects

Each active project is granted a subdirectory under `/projects/<projectname>`. This is where the bulk of data is expected to be, and where jobs should generally be run from. Storage quotas are based on the allocation award.

Quota usage can be viewed at any time by issuing a `cd` command into the project directory, and using the `df -h` command to view total, used, and remaining available space for the mounted project directory.

### NFS Automount System

Project directories are automatically mounted or unmounted via NFS on an "as-needed" basis. /projects directories that have not been accessed for a period of time will be umounted and not immediately visible via a command such as `ls /projects`, but will become immediately available if a file or path is accessed with an `ls`, `cd`, or other file access is made in that path. 

## Home Directories: /home

/home directories are mounted as `/home/<username>`. Home directories are hosted under the user's initial /project directory. Quotas in /home are included as a part of the quota of that project's storage allocation. 

## Scratch Space: /scratch/username and /scratch/username/jobid

For users who also have Eagle allocations, please be aware that scratch space on Swift behaves differently, so adjustments to job scripts may be necessary. 

The scratch directory on each Swift compute node is a 1.8TB spinning disk, and is accessible only on that node. The default writable path for scratch use is `/scratch/<username>`. There is no global, network-accessible `/scratch` space. `/projects` and `/home` are both network-accessible, and may be used as /scratch-style working space instead.


## Temporary space: $TMPDIR 

When a job starts, the environment variable `$TMPDIR` is set to `/scratch/<username>/<jobid>` for the duration of the job. This is temporary space only, and should be purged when your job is complete. Please be sure to use this path instead of /tmp for your tempfiles.

There is no expectation of data longevity in scratch space, and it is subject to purging once the node is idle. If desired data is stored here during the job, please be sure to copy it to a /projects directory as part of the job script before the job finishes.

## Mass Storage System

There is no Mass Storage System for deep archive storage on Swift. However, Swift is expected to be a part of the upcoming Campaign Storage system (VAST storage) in the future, allowing those projects with allocations on Eagle to seamlessly transfer data between clusters, and into the Eagle MSS system.

## Backups and Snapshots

There are no backups or snapshots of data on Swift. Though the system is protected from hardware failure by multiple layers of redundancy, please keep regular backups of important data on Swift, and consider using a Version Control System (such as Git) for important code. 


