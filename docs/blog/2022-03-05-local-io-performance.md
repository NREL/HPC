---
title: Local I/O Performance on Eagle
postdate: March 05, 2022
layout: post
description: A quick analysis of local I/O performance on Eagle compute nodes versus Lustre storage.
---
# Eagle Local I/O Performance

We sometimes receive questions about disk types and I/O performance on compute nodes. Eagle has two network file systems. Qumulo provides /home and /nopt. It is NFS and is not considered a performance file system. /home has snapshots for restoration of lost data, but should not be used as a replacement for a source code repository like Git. Lustre is our performance file system, and it provides storage for the /scratch, /projects, /shared-projects and /datasets directories.

Eagle also has two storage options on the compute nodes. /dev/shm is an in-memory space (shm: shared memory), which is fast, but you need to balance its usage with your job's memory usage as it is located directly in RAM. /tmp/scratch is physical storage. The type of storage and performance differ depending on the specific type of compute node. 

If we look under Eagle's [Compute Node Hardware Details](https://www.nrel.gov/hpc/eagle-system-configuration.html) on the central [NREL HPC website](https://www.nrel.gov/hpc), there are nodes listed as having SATA drives, and nodes listed as having SSDs. Our SATA drives are still spinning disks, while SAS (serial attached SCSI) is how the SSD’s are connected to the node. We would generally expect the nodes with SSDs to perform better. Let’s test that out with a simple test. 

This following is a command we regularly use to verify Lustre OST (object storage target) performance. It’s designed to write enough information so that you are seeing disk performance, and not just the performance of the storage controller of the disk: 

`dd if=/dev/zero of=X bs=1M count=10k`

This is writing in file in chunks of 1M, 10k times, to X. It writes an 11GB file. The results:

*Lustre*: 1.6 GB/s per OST

*Node /dev/shm*: 2.8 GB/s

*Node* SATA spinning disk: 2.4 GB/s

*Node* SAS SSD: 2.4 GB/s

Surprising! There is not a difference between the two local disks. Let’s do the same test, but instead of writing in 1M chunks, we will write in 10M chunks which will write a 107GB file. For this case, Lustre and /dev/shm maintain performance, but here’s what we get for the two local disk types:

*Node* SATA spinning disk: 146 MB/s

*Node* SAS SSD: 1.9 GB/s

That is a rather drastic drop off in performance for the SATA disk. So how your data writes to disk can drastically affect performance. A lot of tiny files will look the same between the two disk types, one large continuous write would differ.
