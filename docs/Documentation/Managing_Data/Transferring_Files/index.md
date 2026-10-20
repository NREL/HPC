---
layout: default
title: File Transfers
grand_parent: Data Movement
parent: Transferring Data
---

# Transferring Files

*Learn how to transfer data within, to and from NLR's high-performance computing (HPC) systems.*

For a video presentation on this topic, please see [Transferring data to and from Kestrel](https://nrel-my.sharepoint.com/:v:/r/personal/chschwin_nrel_gov/Documents/Recordings/Tutorials/Transferring%20Data%20to%20and%20from%20Kestrel.mov?csf=1&web=1&e=hagS2w&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D) by Matt Selensky.

For further information about invidiual systems' filesystem architecture and quotas, please see the [Systems section](../../Systems/). 

## Best Practices for Transferring Files

### File Transfers Between Filesystems on the NLR network

rsync is the recommended tool for transferring data between NLR systems. It allows you to easily restart transfers if they fail, and also provides more consistency when dealing with symbolic links, hard links, and sparse files than either scp or cp. It is recommended you do not use compression for transfers within NLR systems. An example command is:

```bash
$ rsync -aP --no-g /scratch/username/dataset1/ /mss/users/username/dataset1/
```

### Transfers to an External Network

!!! Note
    The [Mass Storage System (MSS)](../mss.md) has quotas that limit the number of individual files you can store. If you are copying hundreds of thousands of files then it is best to archive these files prior to copying to Mass Storage. See the [guide on how to archive files](#archiving-files-and-directories).
    
    Mass Storage quotas rely on the group of the file and not the directory path. It is best to use the `--no-g` option when rsyncing to MSS so you use the destination group rather than the group permissions of your source.  You can also `chgrp` your files to the appropriate group prior to rsyncing to MSS.

#### The Data Management Node (Kestrel only)

[Kestrel](../../Systems/Kestrel/index.md) users are encouraged to transfer data outside of the NLR network through the dedicated data transfer node, `dm1`, whenever possible. The `dm1` node is able to access every filesystem on Kestrel and can be used to initiate large data transfers through any of the methods discussed below without impacting the experience of other users on the cluster.

!!! Note "Accessing the Data Management Node"
    The `dm1` node functions like a login node; users who are on the NLR network may connect via `ssh <username>@dm1.hpc.nlr.gov`. External users can connect via `ssh dm1` from `kestrel.nlr.gov` or use the SSH gateway or HPC VPN. Refer to [Accessing Kestrel](../../Systems/Kestrel/index.md#accessing-kestrel) for more information.

#### Small Transfers (<100GB) outside of the NLR network

`rsync`, `scp`, and `curl` will be your best option for small transfers (<100GB) outside of the NLR network. If your rsync/scp/curl transfers are taking hours to complete then you should consider using [Globus](globus.md), if possible.

If you're transferring many files then you should use rsync:

```bash
$ rsync -aP --no-g /mss/users/username/dataset1/ user@desthost:/home/username/dataset1/
```

!!! Note "Setting Network Bandwidth Limits"
    For transfers over Wi-Fi, you may limit bandwidth with `--bwlimit=RATE`. For example, `--bwlimit=1.5M` sets a limit of 1.5 Megabytes per second.

If you're transferring an individual file then use scp:

```bash
$ scp /home/username/example.tar.gz user@desthost:/home/username/
```

You can use curl or wget to download individual files:
```bash
$ curl -O https://URL
$ wget https://URL
```

Additional rsync examples are available [here](https://github.com/NatLabRockies/HPC/tree/master/general/data-transfer).

#### Large Transfers (>100GB) outside of the NLR network

Globus is optimized for file transfers between data centers and anything outside of the NLR network. It will be several times faster than any other tools you will have available.  See [Transferring Files using Globus](globus.md) for instructions on transferring files with Globus.

#### Transferring files using Windows
For Windows you will need to download WinSCP to transfer files to and from HPC systems over SCP. See the documentation on [Transferring using WinSCP](winscp.md).


## Archiving files and directories

*Learn various techniques to combine and compress multiple files or directories into a single file to reduce storage footprint or simplify sharing.*

### tar

`tar`, along with `zip`, is one of the basic commands to combine multiple individual files into a single file (called a "tarball"). `tar` requires at least one command line option. A typical usage would be:
```bash
$ tar -cf newArchiveName.tar file1 file2 file3
# or
$ tar -cf newArchiveName.tar /path/to/folder/
```

!!! Warning "Disk Usage"
    Please note that this process approximately doubles disk usage of your data, so you should ensure that creating the tarball will not hit the limit of your disk space.

The `-c` flag denotes **c**reating an archive, and `-f` denotes that the next argument given will be the archive name&mdash;in this case it means the name you would prefer for the resulting archive file. 

To extract files from a tar, it's recommended to use:
```bash
$ tar -xvf existingArchiveName.tar
```
`-x` is for **ex**tracting, `-v` uses **v**erbose mode which will print the name of each file as it is extracted from the archive.

### Compressing

`tar` can also generate compressed tarballs which reduce the size of the resulting archive. This can be done with the `-z` flag (which just calls `gzip` on the resulting archive automatically, resulting in a `.tar.gz` extension) or `-j` (which uses `bzip2`, creating a `.tar.bz2`).

For example:

```bash
# gzip
$ tar -czvf newArchive.tar.gz file1 file2 file3
$ tar -xvzf newArchive.tar.gz

# bzip2
$ tar -czjf newArchive.tar.bz2 file1 file2 file3
$ tar -xvjf newArchive.tar.bz2
```
