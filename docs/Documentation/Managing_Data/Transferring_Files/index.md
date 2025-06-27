---
layout: default
title: File Transfers
grand_parent: Data Movement
parent: Transferring Data
---

# Transferring Files

*Learn how to transfer data within, to and from NREL's high-performance computing (HPC) systems.*

For a video presentation on this topic, please see [Transfering data to and from Kestrel](https://nrel-my.sharepoint.com/:v:/r/personal/chschwin_nrel_gov/Documents/Recordings/Tutorials/Transferring%20Data%20to%20and%20from%20Kestrel.mov?csf=1&web=1&e=hagS2w&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D) by Matt Selensky.

For further information about invidiual systems' filesystem architecture and quotas, please see the [Systems section](../../Systems/). 

## Best Practices for Transferring Files

#### File Transfers Between Filesystems on the NREL network

rsync is the recommended tool for transferring data between NREL systems. It allows you to easily restart transfers if they fail, and also provides more consistency when dealing with symbolic links, hard links, and sparse files than either scp or cp. It is recommended you do not use compression for transfers within NREL systems. An example command is:

```bash
$ rsync -aP --no-g /scratch/username/dataset1/ /mss/users/username/dataset1/
```

*Mass Storage has quotas that limit the number of individual files you can store. If you are copying hundreds of thousands of files then it is best to archive these files prior to copying to Mass Storage. See the [guide on how to archive files](#archiving-files-and-directories).*

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

Additional rsync examples are available [here](https://github.com/NREL/HPC/tree/master/general/data-transfer).

#### Large Transfers (>100GB) outside of the NREL network

Globus is optimized for file transfers between data centers and anything outside of the NREL network. It will be several times faster than any other tools you will have available. Documentation about requesting a HPC Globus account is available on the [Globus Services page on the HPC website](https://www.nrel.gov/hpc/globus-file-transfer.html).  See [Transfering files using Globus](globus.md) for instructions on transfering files with Globus.

#### Transfering files using Windows
For Windows you will need to download WinSCP to transfer files to and from HPC systems over SCP. See [Transfering using WinSCP](https://www.nrel.gov/hpc/winscp-file-transfer.html).


## Archiving files and directories

*Learn various techniques to combine and compress multiple files or directories into a single file to reduce storage footprint or simplify sharing.*

### tar

`tar`, along with [`zip`](#zip), is one of the basic commands to combine multiple individual files into a single file (called a "tarball"). `tar` requires at least one command line option. A typical usage would be:
```bash
$ tar -cf newArchiveName.tar file1 file2 file3
# or
$ tar -cf newArchiveName.tar /path/to/folder/
```

The `-c` flag denotes **c**reating an archive, and `-f` denotes that the next argument given will be the archive name&mdash;in this case it means the name you would prefer for the resulting archive file. 

To extract files from a tar, it's recommended to use:
```bash
$ tar -xvf existingArchiveName.tar
```
`-x` is for **ex**tracting, `-v` uses **v**erbose mode which will print the name of each file as it is extracted from the archive.

### Compressing

`tar` can also generate compressed tarballs which reduce the size of the resulting archive. This can be done with the `-z` flag (which just calls [`gzip`](#gzip) on the resulting archive automatically, resulting in a `.tar.gz` extension) or `-j` (which uses [`bzip2`](#bzip2), creating a `.tar.bz2`).

For example:

```bash
# gzip
$ tar -czvf newArchive.tar.gz file1 file2 file3
$ tar -xvzf newArchive.tar.gz

# bzip2
$ tar -czjf newArchive.tar.bz2 file1 file2 file3
$ tar -xvjf newArchive.tar.bz2
```
