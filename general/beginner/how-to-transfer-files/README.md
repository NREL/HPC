# how-to-transfer-files

*Learn how to transfer data within, to and from NREL's high-performance computing (HPC) systems.*

A supported set of instructions for data transfer using NREL HPC systems is provided on the [HPC NREL Website](https://www.nrel.gov/hpc/data-storage-transfer.html).

## Checking Usage and Quota
The below command is used to check your quota from a Peregrine login node.  alloc_tracker will display your usage and quota for each filesystem.

```bash
$ alloc_tracker
```

## Best Practices for Transfering Files

#### File Transfers Between Filesystems on the NREL network

rsync is the recommended tool for transferring data between NREL systems. It allows you to easily restart transfers if they fail, and also provides more consistency when dealing with symbolic links, hard links, and sparse files than either scp or cp. It is recommended you do not use compression for transfers within NREL systems. An example command is:

```bash
$ rsync -aP --no-g /scratch/username/dataset1/ /mss/users/username/dataset1/
```

*Mass Storage has quotas that limit the number of individual files you can store. If you are copying hundreds of thousands of files then it is best to archive these files prior to copying to Mass Storage. See the [guide on how to archive files](../intro-to-linux/archiving.md).*

*Mass Storage quotas rely on the group of the file and not the directory path. It is best to use the `--no-g` option when rsyncing to MSS so you use the destination group rather than the group permissions of your source.  You can also `chgrp` your files to the appropriate group prior to rsyncing to MSS.*

#### Small Transfers (<100GB) outside of the NREL network
`rsync`, `scp`, and `curl` will be your best option for small transfers (<100GB) outside of the NREL network. If your rsync/scp/curl transfers are taking hours to complete then you should consider using [Globus](globus.md).

If you're transferring many files then you should use rsync:

```bash
$ rsync -azP --no-g /mss/users/username/dataset1/ user@desthost:/home/username/dataset1/
```

If you're transferring an individual file then use scp:

```bash
$ scp /home/username/example.tar.gz user@desthost:/home/username/
```

You can use curl or wget to download individual files:
```bash
$ curl -O https://URL
$ wget https://URL
```

#### Large Transfers (>100GB) outside of the NREL network

Globus is optimized for file transfers between data centers and anything outside of the NREL network. It will be several times faster than any other tools you will have available. Documentation about requesting a HPC Globus account is available on the [Globus Services page on the HPC website](https://www.nrel.gov/hpc/globus-file-transfer.html).  See [Transfering files using Globus](globus.md) for instructions on transfering files with Globus.

#### Transfering files using Windows
For Windows you will need to download WinSCP to transfer files to and from HPC systems over SCP. See [Transfering using WinSCP](winscp.md).
