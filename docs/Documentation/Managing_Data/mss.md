# Mass Storage Sytem (MSS)
*NREL’s Amazon Web Services (AWS) Mass Storage System (MSS)
is an additional data archival resource available to active projects and users
on the Kestrel high-performance computing (HPC) system.*

The AWS MSS keeps and protects important data, primarily as an addition to
Kestrel's high-performance Lustre filesystem (/projects and /scratch).

NREL implemented the AWS MSS to take advantage of S3 Deep Glacier archiving,
replacing the previous on-premises MSS, Gyrfalcon, which reached end-of-life at
the end of 2020. 

##How To Copy/Move Data from Kestrel 

AWS charges per inode.  Therefore, to keep costs down it is recommended 
users create a compressed tarball of any files and/or directories desired 
to be archived to AWS MSS.  The size limit per archived file is 5TB, and therefore
individual tarballs need to be under this limit (although multiple tarballs that sum to greater than 5 TB can be archived).  

**The recommended command is:**

```$ tar czvf /destination/descriptor-YYYMMDD.tgz <source-files-directories\>```

**Example, from Kestrel’s /projects/csc000/data-to-be-copied from a Kestrel Login
node:**

```
$ cd /projects/csc000
$ tar czvf /kfs2/shared-projects/MSS/projects/csc000/data-to-be-copied-20211215.tgz data-to-be-copied
```

Data placed in ```/kfs2/shared-projects/MSS/projects/<project_handle>``` and
```/kfs2/shared-projects/MSS/home/<username>``` is synced to AWS MSS and then purged from Kestrel.

##How To Restore Data 

- Restore requests of AWS MSS data will require a request to
the [HPC Help Desk](mailto:HPC-Help@nrel.gov) and may require 48 hours or more to be able to stage from
Deep Archive to recover.  
- Users can see a list of the archived files they have on AWS MSS by searching the following file: ```/kfs2/shared-projects/MSS/MSS-archived-files```
    - The MSS-archived-files has limited information, but all archives 
      related to a project can be found using a command such as:
      ```$ grep <project name> /kfs2/shared-projects/MSS/MSS-archived-files```

- Let the [HPC Help Desk](mailto:HPC-Help@nrel.gov) know specifically what file(s) you would like to recover, and where the
recovered files should be placed.  

##Usage Policies 
Follow the [AWS MSS policies](https://www.nrel.gov/hpc/mass-storage-system-policies.html).

##Contact 
Contact the [HPC Help Desk](mailto:HPC-Help@nrel.gov) if you have any questions or issues.
