---
layout: default
title: Running Multiple Sub-Jobs
has_children: false
---

# Running Multiple Sub-Jobs with One Job Script 

If your workload consists of serial or modestly parallel programs, you can run multiple instances of your program at the same time using different processor cores on a single node. This will allow you to make better use of your allocation because it will use the resources on the node that would otherwise be idle.

## Example

For illustration, we use a simple C code to calculate pi. The source code and instructions for building that program are provided below:

### Sample Program

Copy and paste the following into a terminal window that's connected to the cluster.
This will stream the pasted contents into a file called `pi.c` using the command `cat << eof > pi.c`.

```c
cat << eof > pi.c
#include <stdio.h>

// pi.c: A sample C code calculating pi

main() {
  double x,h,sum = 0;
  int i,N;
  printf("Input number of iterations: ");
  scanf("%d",&N);
  h=1.0/(double) N;

  for (i=0; i<N; i++) {
   x=h*((double) i + 0.5);
   sum += 4.0*h/(1.0+x*x);
  }

  printf("\nN=%d, PI=%.15f\n", N,sum);
}

eof
```

### Compile the Code 

This example uses the Intel C compiler. Load the module and compile pi.c with the following commands:

```
$ module purge
$ module load intel-mpi
$ icc -O2 pi.c -o pi_test
$ ./pi_test
```

A sample batch job script file to run 8 copies of the pi_test program on a node with 24 processor cores is given below. This script creates 8 directories and starts 8 jobs, each in the background. It waits for all 8 jobs to complete before finishing.

### Copy and paste the following into a text file 

Place that batch file into one of your directories on the cluster. Make sure to change the allocation to a project-handle you belong to.

```bash
#!/bin/bash
## Required Parameters   ##############################################
#SBATCH --time 10:00               # WALLTIME limit of 10 minutes

## Double ## will cause SLURM to ignore the directive:
#SBATCH -A <handle>                # Account (replace with appropriate)

#SBATCH -n 8                       # ask for 8 tasks   
#SBATCH -N 1                       # ask for 1 node
## Optional Parameters   ##############################################
#SBATCH --job-name wait_test       # name to display in queue
#SBATCH --output std.out
#SBATCH --error std.err

JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above

# Run 1 job per task
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks

for((i=1;i<=$N_JOB;i++))
do
  mkdir $JOBNAME.run$i             # Make subdirectories for each job
  cd $JOBNAME.run$i                # Go to job directory
  echo 10*10^$i | bc > input       # Make input files
  time ../pi_test < input > log &  # Run your executable, note the "&"
  cd ..
done

#Wait for all
wait

echo
echo "All done. Checking results:"
grep "PI" $JOBNAME.*/log

```

### Submit the Batch Script

Use the following Slurm sbatch command to submit the script. The job will be scheduled, and you can view the output once the job completes to confirm the results.

`$ sbatch -A <project-handle> <batch_file>`

