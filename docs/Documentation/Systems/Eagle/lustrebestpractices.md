---
layout: default
title: Filesystems
parent: Lustre
grand_parent: Filesystems and I/O
---
# Eagle Filesystems

## Home File System

The Home File System (HFS) subsystem on Eagle is a robust NFS file system intended to provide highly reliable storage for user home directories and NREL-specific software. HFS has a capacity of 182 TB. Snapshots (backup copies) of files in the HFS filesystem are available up to 30 days after change/deletion.

**/home**

The /home directory on Eagle resides on HFS and is intended to hold small files. These include shell startup files, scripts, source code, executables, and data files.  Each user has a quota of 50 GB.

**/nopt**

The /nopt directory on Eagle resides on HFS and is where NREL-specific software, module files, licenses, and licensed software is kept.

## Parallel File System

The Parallel File System (PFS) on Eagle is a parallel Lustre file system intended for high-performance I/O.  Use PFS storage for running jobs and any other intensive I/O activity. The capacity of 17 PB is provided by 36 Object Storage Servers (OSSs) and 72 Object Storage Targets (OSTs) with 3 Metadata Servers, all connected to Eagle's Infiniband network with 100 Gb/sec EDR. The default stripe count is 1, and the default stripe size is 1 MB.

The PFS hosts the /scratch, /projects, /shared-projects, and /datasets directory.

**There are no backups of PFS data.**  Users are responsible for ensuring that critical data is copied to [Mass Storage](https://www.nrel.gov/hpc/mass-storage-system.html) or other alternate data storage location.

**/scratch**

Each user has their own directory in /scratch. Data in /scratch is subject to deletion after 28 days of inactivity.

**/projects**

Each project/allocation has a directory in /projects intended to host data, configuration, and applications shared by the project.

**/shared-projects**

Projects may request a shared project directory to host data, configuration, and applications shared by multiple projects/allocations.

**/datasets**

The /datasets directory on Eagle hosts widely used data sets. 

There are multiple big data sets that are commonly used across various projects for computation and analysis on NREL's HPC Systems. We provide a common location on Eagle's scratch filesystem at /datasets, where these data sets are available for global reading by all compute nodes on Eagle. Each data set contains a readme file that covers background, references, explanation of the data structure, and Python examples.

**/datasets/NSRDB**

The National Solar Radiation Database (NSRDB) is a serially complete collection of meteorological and solar irradiance data sets for the United States and a growing list of international locations for 1998-2017. The NSRDB provides foundational information to support U.S. Department of Energy programs, research, and the general public.

**/datasets/WIND**

The Wind Integration National Data Set (WIND) Toolkit consists of wind resource data for North America and was produced using the Weather Research and Forecasting Model (WRF).

## Node File System

Each Eagle compute node has a local solid-state drive (SSD) for use by compute jobs. They vary in size; 1 TB (standard), 1.6 TB (bigmem), and 25.6 TB (bigscratch), depending on the node feature requested. There are several possible scenarios in which a local disk may make your job run faster. For instance, you may have a job accessing or creating many small (temporary) files, you may have many parallel tasks accessing the same file, or your job may do many random reads/writes or memory mapping.

**/tmp/scratch**

The local disk is mounted at /tmp/scratch and set under the $LOCAL_SCRATCH environment variable during a job. A node will not have read or write access to any other node's local scratch, only its own. Also, this directory will be cleaned once the job ends. You will need to transfer any files to be saved to another file system. 

For more information about requesting this feature, please see Resource Request Descriptions on the Eagle Batch Jobs page.

## Lustre Best Practices
In some cases special care must be taken while using Lustre so as not to affect the performance of the filesystem for yourself and other users. The below Do's and Don'ts are provided as guidance. 

* **Do**
    * Use the `lfs find`
        * e.g. 
        ```shell
        lfs find /scratch/username -type f -name "*.py"
        ```
    * Break up directories with many files into more directories if possible
    * Store small files and directories of small files on a single OST (Object Storage Target) 
    * Limit the number of processes accessing a file. It may be better to read in a file once and then broadcast necessary information to other processes
    * Change your stripecount based on the filesize
    * Write many files to the node filesystem `/tmp/scratch/`: this is local storage on each node, and is not a part of the Lustre filesystem. Once your work is complete, the files can then be added to a tar archive and transferred to the `/project/project_name` for later use, or deleted from /tmp/scratch if no longer needed
    * Store data and run executables from `/projects`
        * Storing your conda environments in `/projects` can ensure that your data and executables are on the same filesystem, improving performance

* **Do Not**
    * Use `ls -l`
    * Have a file accessed by multiple processes
    * In Python, avoid using `os.walk` or `os.scandir`
    * List files instead of using wildcards
        * e.g. don't use `cp * dir/`
        * If you need to tar/rm/cp a large number of files use xargs or similar:
        ```shell
        lfs find /scratch/username/old_data/ -t f -print0 | xargs -0 rm
        ```
    * Have many small files in a single directory
    * Store important files in `/scratch`
        * e.g. don't keep data, libraries or programs in `/scratch/username`, as `/scratch` directories are subject to automated purging based on the [Data Retention Policy](https://www.nrel.gov/hpc/data-retention-policy.html)


## Useful Lustre commands

* Check your storage usage:
    * `lfs quota -h -u <username> /scratch`
* See which MDT a directory is located on
    * `lfs getstripe --mdt-index /scratch/<username>`
    * This will return an index 0-2 indicating the MDT
* Create a folder on a specific MDT (admin only)
    * `lfs mkdir –i <mdt_index> /dir_path`

## Striping

Lustre provides a way to stripe files, this spreads them across multiple OSTs. Striping a large file being accessed by many processes can greatly improve the performace. See [Lustre file striping](http://wiki.lustre.org/Configuring_Lustre_File_Striping) for more details. 

```
lfs setstripe <file> -c <count> -s <size>
```
* The stripecount determines how many OST the data is spread across
* The stripe size is how large each of the stripes are in KB, MB, GB

## References
* [Lustre manual](http://doc.lustre.org/lustre_manual.xhtml)
* [CU Boulder - Lustre Do's and Don'ts](http://researchcomputing.github.io/meetup_fall_2014/pdfs/fall2014_meetup10_lustre.pdf)
* [NASA - Lustre Best Practices](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html)
* [NASA - Lustre basics](https://www.nas.nasa.gov/hecc/support/kb/lustre-basics_224.html)
* [UMBC - Lustre Best Practices](https://hpcf.umbc.edu/general-productivity/lustre-best-practices/)
* [NICS - I/O and Lustre Usage](https://www.nics.tennessee.edu/computing-resources/file-systems/io-lustre-tips)
* [NERSC - Lustre](https://docs.nersc.gov/performance/io/lustre/)
