---
layout: default
title: Filesystems
parent: Systems
has_children: true
---



# Filesystems on Swift

Swift's central storage consists of approximately 3PB of NFS-mounted storage, served over NFS (Network File System). The underlying filesystem management is via ZFS. 

Unlike the Lustre-based storage on Eagle, Swift currently does not host a parallel file system.

## Project Storage: /projects

Each active project is granted a subdirectory under `/projects/<projectname>`. This is where the bulk of data is expected to be, and where jobs should generally be run from. Storage quotas are based on the allocation award.

Quota usage can be viewed at any time by issuing a `cd` command into the project directory, and using the `df -h` command to view total, used, and remaining available space for the mounted project directory.

### NFS Automount System

Project directories are automatically mounted or unmounted via NFS on an "as-needed" basis. /projects directories that have not been accessed for a period of time will be umounted and not immediately visible via a command such as `ls /projects`, but will become immediately available if a file or path is accessed with an `ls`, `cd`, or other file access is made in that path. 

## Home Directories: /home

/home directories are mounted as `/home/<username>`. Home directories are hosted under the user's initial /project directory. Quotas in /home are included as a part of the quota of that project's storage allocation. 

## Scratch Space: /scratch/username and /scratch/username/jobid

For users who also have Eagle accounts, please be aware that `/scratch` on Swift behaves differently, so adjustments to job scripts may be necessary.

There is NO global/network-accessible `/scratch` path on Swift at this time. If a networked working directory is needed for multi-node jobs, please use /projects. 

The scratch directory on each Swift compute node is a 1.8TB spinning disk, and is accessible only on that node. The default writable path for scratch use is `/scratch/<username>`. 

##
When a job starts, the environment variable `$TMPDIR` is set to `/scratch/<username>/<jobid>` for the duration of the job. This is temporary space only, and should be purged when your job is complete. 

There is no expectation of data longevity in scratch space, and it is subject to purging once the node is idle. If desired data is stored here during the job, please be sure to copy it to a /projects directory as part of the job script before the job finishes.



