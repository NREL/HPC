# Archiving and Compressing Multiple Files and/or Directories

*Learn various techniques to combine and compress multiple files or directories into a single file to reduce storage footprint or simplify sharing.*

## *This is a temporary file to provide verbosity, edit/remove it or its content as needed.*

## `tar`

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

## `gzip`

## `bzip2`

## `zip`

## Splitting




