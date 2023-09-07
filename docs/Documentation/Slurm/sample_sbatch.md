---
layout: default
title: Example SBATCH Scripts
has_children: false
---

# Example SBATCH Script Walkthrough
The primary method of submitting an unattended job to the Slurm scheduler queue is via a batch script. 

Many examples of sbatch scripts are available in the [HPC Repository Slurm Directory](https://github.com/NREL/HPC/tree/master/slurm) on Github. 

Here's an example script to get started. These scripts may be adapted to any HPC system with minor modifications.

```
#!/bin/bash
#SBATCH --account=<allocation>
#SBATCH --time=4:00:00
#SBATCH --job-name=job
#SBATCH --mail-user=your.email@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID
module load myprogram
myprogram.sh
```

## Script Details

Here is a section-by-section breakdown of the sample sbatch script, to help you begin writing your own.

### Script Begin

`#!/bin/bash`

This denotes the start of the script, and that it is written in BASH shell language, the most common Linux environment. 

### SBATCH Directives

```
#SBATCH --account=<allocation>
#SBATCH --time=4:00:00
#SBATCH --job-name=job
#SBATCH --mail-user=your.email@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=job_output_filename.%j.out  # %j will be replaced with the job ID
```
Generalized form:

`#SBATCH --<command>=<value>` 

Command flags to the sbatch program are given via `#SBATCH` directives in the sbatch script. There are many flags available that can affect your job. See the official [Slurm documentation on sbatch](https://slurm.schedmd.com/sbatch.html) for a complete list, or view the man page on a login node with `man sbatch`. 

Sbatch directives must be at the beginning of your sbatch script. Once a line with any other non-directive content is detected, Slurm will no longer parse further directives.

Note that sbatch flags do not need to be issued via directives inside the script. They can also be issued via the commandline when submitting the job. Flags issued via commandline will supercede directives issued inside the script. For example:

`sbatch --account=csc000 --time=60 --partition=debug mytestjob.sh`

#### Job Instructions

After the sbatch directive block, you may then begin executing your job. The syntax is normal BASH shell scripting. You may load system modules for software, load virtual environments, define environment variables, and execute your software to perform work. 

In the simplest form, your sbatch script should load your software module(s) required, and then execute your program. 

```
module load myprogram
srun myprogram.sh
```
or

```
module load myprogram
myprogram.sh
```

You may also use more advanced bash scripting as a part of your sbatch script, e.g. to set up environments, manage your input and output files, and so on.

More system-specific information about Slurm partitions, node counts, memory limits, and other details can be found under the appropriate [Systems](../Systems/index.md) page.

See the "master" main branch of the [Github repository](https://www.github.com/NREL/HPC) for downloadable examples.




