# Lustre Best Practices
In some cases special care must be taken while using Lustre so as not to affect the performance of the filesystem for yourself and other users. The below Do's and Don'ts are provided as guidance. 

## Do
* Use the `lfs find`
    * e.g. 
    ```shell
    lfs find /scratch/username -type f -name "*.py"
    ```
* Break up directories with many files into more directories if possible
* Store small files and directories of small files on a single OST (Object Storage Target) 
* Limit the number of processes accessing a file. It may be better to read in a file once and then broadcast necessary information to other processes
* Change your stripecount based on the filesize
* Write many files to the node filesystem `$TMPDIR`: some compute nodes have local storage available, and it is not a part of the Lustre filesystem. Once your work is complete, the files can then be added to a tar archive and transferred to the `/project/project_name` for later use, or deleted from `$TMPDIR` if no longer needed.
* Store data in `/projects` and run jobs from `/scratch/$USER`
* Storing your conda environments in `/projects` or `/scratch`

## Do Not
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
