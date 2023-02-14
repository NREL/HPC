---
layout: default
title: Acquire
parent: Building Packages
grand_parent: Intermediate
---

## Getting the package

1. Change working directory to the location where you'll build the package. A convenient location is `/scratch/$USER`, which we'll use for this example. `cd /scratch/$USER`

2. OpenMPI can be found at [https://www.open-mpi.org/software/ompi/](https://www.open-mpi.org/software/ompi/). This will automatically redirect you to the latest version, but older releases can be seen in the left menu bar. For this, choose version 4.1.

3. There are several packaging options. Here, we'll get the bzipped tarball `openmpi-4.1.0.tar.bz2`. You can either download it to a local machine (laptop) and then scp the file over to Eagle, or get it directly on Eagle with wget.
```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.bz2
```

    You should now have a compressed tarball in your scratch directory.

4. List the contents of the tarball before unpacking. This is very useful to avoid inadvertently filling a directory with gobs of files and directories when the tarball has them at the top of the file structure),
```
tar -tf openmpi-4.1.0.tar.bz2
```

5. Unpack it via
```
tar -xjf openmpi-4.1.0.tar.bz2
```
If you're curious to see what's in the file as it unpacks, add the `-v` option. 

6. You should now have an `openmpi-4.1.0` directory. `cd openmpi-4.1.0`, at which point you are in the top level of the package distribution.

You can now proceed to configuring, making, and installing.

