---
title: March 2022 Monthly Update
data: 2022-03-02
layout: default
brief: April Outage, Standby QOS, IO Performance
---
# March HPC Systems Power Outage
Eagle, Swift, Vermillion, Meridian, and all other related HPC systems, services, and filesystems will be unavailable beginning on Thursday, March 31st, 2022 due to scheduled facilities maintenance. The Eagle Operations Team will also be performing firmware updates, security patches, and updates to the GPU node images during this time. The outage is anticipated to start at 7:00am on Thursday, March 31st and last at least through Friday, April 1st, 2022, but it may extend through to the following Monday, April 4th, 2022. We will provide updates as more information about the full outage period becomes available.

# Standby QOS now Available on Eagle & Swift
Through the annual user survey and direct feedback, we have received requests to submit jobs to the standby queue on demand. We are now pleased to announce the capability to submit jobs to the standby Quality of Service (QoS) queue by using the `--qos=standby` flag. Please remember that standby jobs run with a very low priority, so these jobs may wait for a considerable period of time. However, the job will not incur any AU charges against your project's allocation.

# Eagle local I/O performance
We’ve received questions recently about disk types and performance on compute nodes. Eagle has two network file systems. Qumulo provides /home and /nopt. It is NFS and is not considered a performance file system. /home has snapshots for restoration of lost data, but should not be used as a replacement for a source code repository like Git. Lustre is our performance file system and provides /scratch, /projects, /shared-projects and /datasets.

Eagle also has two options on nodes. /dev/shm, which is an in-memory space, which is fast but you need to balance its usage with your jobs memory usage. /tmp/scratch is physical storage. The type of storage and performance differ depending on node and that’s what we hope to clarify. If we look under [Compute Node Hardware Details](https://www.nrel.gov/hpc/eagle-system-configuration.html) on the central [NREL HPC website](https://hpc.nrel.gov), there are nodes listed as having SATA drives, and nodes listed as having SSDs. Our SATA drives are still spinning disks, while SAS (serial attached SCSI) is how the SSD’s are connected to the node. We would generally expect the nodes with SSDs to perform better. Let’s test that out with a simple test. This is a command we regularly use to verify Lustre OST (object storage target) performance. It’s designed to write enough information so that you are seeing disk performance, and not just the performance of the storage controller of the disk: dd if=/dev/zero of=X bs=1M count=10k

This is writing in file in chunks of 1M, 10k times, to X. It writes an 11GB file:

Lustre: 1.6 GB/s per OST

Node /dev/shm: 2.8 GB/s

Node SATA spinning disk: 2.4 GB/s

Node SAS SSD: 2.4 GB/s

Surprising! There is not a difference between the two local disks. Let’s do the same test, but instead of writing in 1M chunks, we will write in 10M chunks which will write a 107GB file. For this case, Lustre and /dev/shm maintain performance, but here’s what we get for the two local disk types:

Node SATA spinning disk: 146 MB/s

Node SAS SSD: 1.9 GB/s

That is a rather drastic drop off in performance for the SATA disk. So how your data writes to disk can drastically affect performance. A lot of tiny files will look the same between the two disk types, one large continuous write would differ.


