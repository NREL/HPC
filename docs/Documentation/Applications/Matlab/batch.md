# Running MATLAB in Batch Mode

*Learn how to run MATLAB software in batch mode on the Eagle system*

Below is an example MATLAB script, matlabTest.m, that creates and populates a
vector using a simple for-loop and writes the result to a binary file,
x.dat. The shell script matlabTest.sb can be passed to the scheduler to run the
job in batch (non-interactive) mode.

To try the example out, create both matlabTest.sb and matlabTest.m files in an
appropriate directory, `cd` to that directory, and call sbatch:

```bash
$ sbatch matlabTest.sb
```

!!! note

    Note: MATLAB comprises many independently licensed components, and in your work
    it might be necessary to wait for multiple components to become
    available. Currently, the scheduler does not handle this automatically. Because
    of this, we strongly recommend using compiled MATLAB code for batch processing.

Calling `squeue` should show that your job is queued:

```
JOBID       PARTITION       NAME       USER       ST       TIME       NODES       NODELIST(REASON)
<JobID>     <partition>     matlabTe   username   PD       0:00       1           (<reason>)
```

<!-- TODO: Are the stdout and stderr filenames correct? -->

Once the job has finished, the standard output is saved in a file called
`test_matlab.o<JobID>`, standard error to `test_matlab.e<JobID>`, and the binary
file `x.dat` contains the result of the MATLAB script.

## Notes on matlabTest.sb File

<!-- TODO: Update User Accounts link in 2nd bullet below -->

- Setting a low walltime increases the chances that the job will be scheduled
  sooner due to backfill.
- The `--account=<account_string>` flag must include a valid account string or
  the job will encounter a permanent hold (it will appear in the queue but will
  never run).  For more information, see [user
  accounts](https://www.nrel.gov/hpc/user-accounts.html).
- The environment variable `$SLURM_SUBMIT_DIR` is set by the scheduler to the
  directory from which the sbatch command was executed, e.g., `/scratch/$USER.`
  In this example, it is also the directory into which MATLAB will write the
  output file x.dat.
  
**matlabTest.sb**

```bash
#!/bin/bash  --login
#SBATCH --time=05:00          # Maximum time requested for job (5 min.)
#SBATCH --nodes=1                # Number of nodes
#SBATCH --job-name=matlabTest              # Name of job
#SBATCH --account=<account_string>        # Program-based WCID (account string associated with job)

module purge                    
module load matlab/R2018b 
  
# execute code  
cd $SLURM_SUBMIT_DIR                        # Change directories (output will save here)
matlab -nodisplay -r matlabTest          # Run the MATLAB script  
```

**matlabTest.m**

```matlab
format long
xmin = 2;
xmax = 10;
x = zeros(xmax-xmin+1,1);
for i = xmin:xmax
    display(i);
    x(i-xmin+1) = i
end
savefile = 'x.dat';
save(savefile,'x','-ASCII')
exit
```
