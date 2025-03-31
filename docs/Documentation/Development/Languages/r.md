# Running R Statistical Computing Environment Software

*Learn how to run the R statistical computing environment software.*

## What Is R?

R is an open-source programming language designed for statistical computing and graphics. It is the current standard for the development of new statistical methodologies and enjoys a large user base.

For more information related to the R project, see the [R website](http://www.r-project.org/).

## Accessing R

The supported method for using R on NREL HPC systems is via Anaconda/mamba. In order to install R, first load the mamba module. On Kestrel or Swift, this is `module load mamba`. Next, create a new conda environment that contains at least the `r-base` package, which installs R itself. Optionally, install the `r-essentials` bundle, which provides many of the most popular R packages for data science, such as the [tidyverse](https://www.tidyverse.org) family of packages.

For example, to create and activate a new environment named `r_env` in your current directory that includes the `r-essentials` bundle:

```bash
module load mamba
conda create --prefix=./r_env r-essentials r-base
conda activate ./r_env
```

!!! note
    We install the `r_env` conda environment in our current directory with the `--prefix` option for a number of reasons. Note that if you create an environment with the `-n` or `--name` option, it will install into your home directory by default, which is not ideal due to its limited storage. The `--prefix` option can also accept an absolute path, such as a dedicated `/projects` directory. Please see our dedicated [conda documentation](../../Environment/Customization/conda.md#creating-environments-by-location) for more information.

For more information about using R in the Anaconda framework, see [Using R language with Anaconda](https://docs.anaconda.com/free/anaconda/packages/using-r-language/).

## Running R Interactively

R is most commonly used via an interactive shell. To do this, first request an interactive compute node ([see running interactive jobs](../../Slurm/interactive_jobs.md)) using the `salloc` command. Alternatively, R can be used through [Jupyterhub](../Jupyter/index.md).

Once on a compute node, R environments can be accessed through Anaconda as described above. To access the R interactive console, type R at the command line. You will be prompted with the familiar R console in your terminal window: 

??? note "R Terminal"

    ```
    $ R

    R version 4.4.2 (2024-10-31) -- "Pile of Leaves"
    Copyright (C) 2024 The R Foundation for Statistical Computing
    Platform: x86_64-conda-linux-gnu

    R is free software and comes with ABSOLUTELY NO WARRANTY.
    You are welcome to redistribute it under certain conditions.
    Type 'license()' or 'licence()' for distribution details.

    Natural language support but running in an English locale

    R is a collaborative project with many contributors.
    Type 'contributors()' for more information and
    'citation()' on how to cite R or R packages in publications.

    Type 'demo()' for some demos, 'help()' for on-line help, or
    'help.start()' for an HTML browser interface to help.
    Type 'q()' to quit R.
    ```

!!! note
    You can run individual R commands directly from the command line via `R -e 'COMMAND'`, such as `R -e 'print("Hello, world")'`.

## Running R Scripts

Since running R programs line by line in the interactive console can be a little tedious, it is often better to combine R commands into a single script and have R execute them all at once. R scripts are text files containing R commands with file extension .R: 

!!! example "hello_world.R"

    ```
    message = "Hi there!"
    nums = sample(1:100, 5)
    cat(message, "\n")
    cat("Here are some random numbers: ", paste(nums, sep = ", "),"\n")
    ```

There are several options for running R scripts:

??? example "source()"

    The source()  function will execute R scripts from inside the interactive console.
    
    ```
    > source("hello_world.R")
      Hi there! 
      Here are some random numbers:  100 41 14 82 63 
    ```

??? example "Rscript"

    The Rscript command can be used to run R scripts from the command line. Output is piped to the stdout.

    ```
    $ Rscript hello_world.R
    Hi there! 
    Here are some random numbers:  71 37 50 24 90 
    ```
    
??? example "R CMD BATCH"

    R CMD BATCH is an older function that behaves similar to Rscript. All output is piped to a corresponding .Rout file.

    ```
    $ R CMD BATCH --no-site-file hello_world.R
    $ cat hello_world.Rout 

    > #hello_world.R
    > 
    > message = "Hi there!"
    > nums = sample(1:100, 5)
    > cat(message, "\n")
    Hi there! 
    > cat("Here are some random numbers: ", paste(nums, sep = ", "),"\n")
    Here are some random numbers:  41 51 61 70 43 
    > 
    > proc.time()
       user  system elapsed 
      0.188   0.024   0.277 
    ```
    
## Submitting Jobs

Another option for using R on the HPC systems is to submit batch jobs to be run on non-interactive nodes. An example job script for running the hello_world.R example is below. Ensure you update your allocation name as well as the path of the conda environment where R has been installed. 

!!! note
    The following example script is submitted to the [shared partition](../../Systems/Kestrel/Running/index.md#shared-node-partition) on Kestrel, which allows nodes to run multiple jobs at once. This is because without invoking any special parallel packages in the `hello_world.R` script, R is limited to using a single core. Refer to the [Parallel Programming in R](#parallel-programming-in-r) section below for information on how to use more than one core from your R code.

```bash
#! /bin/bash
#SBATCH --job-name=helloworld
#SBATCH --nodes=1
#SBATCH --partition=shared
#SBATCH -n 1 
#SBATCH -N 1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --account=<your_allocation_id>
  
module load mamba
conda activate /path/to/r_env
Rscript hello_world.R
```
    
## Versions and Packages

R is a popular open-source language with an active development community. New versions of R are frequently released. Any version can be installed into a custom anaconda environment. Commands for using other versions is shown below: 

??? example "Custom Installation with Conda"

    ```
    $ mamba search r-essentials
    Loading channels: done
    # Name                       Version           Build  Channel
    r-essentials                   1.5.2        r3.3.2_0  conda-forge
    r-essentials                   3.4.1        r3.4.1_0  conda-forge
    r-essentials                   3.5.1       r351_1000  conda-forge
    r-essentials                   3.5.1       r351_2000  conda-forge
    r-essentials                   3.5.1        r35_2001  conda-forge
    r-essentials                     3.6        r36_2001  conda-forge
    r-essentials                     3.6        r36_2002  conda-forge
    r-essentials                     4.0        r40_2002  conda-forge
    r-essentials                     4.0 r40hd8ed1ab_2002  conda-forge
    r-essentials                     4.1 r41hd8ed1ab_2002  conda-forge
    r-essentials                     4.1 r41hd8ed1ab_2003  conda-forge
    r-essentials                     4.2 r42hd8ed1ab_2003  conda-forge
    r-essentials                     4.2 r42hd8ed1ab_2004  conda-forge
    r-essentials                     4.3 r43hd8ed1ab_2004  conda-forge
    r-essentials                     4.3 r43hd8ed1ab_2005  conda-forge
    r-essentials                     4.4 r44hd8ed1ab_2005  conda-forge
    
    $ mamba create --prefix=./test-R r-essentials==4.3
    <Text>
    $ conda activate ./test-R
    (test-R) $ R --version
    R version 4.4.2 (2024-10-31) -- "Pile of Leaves"
    Copyright (C) 2024 The R Foundation for Statistical Computing
    Platform: x86_64-conda-linux-gnu

    R is free software and comes with ABSOLUTELY NO WARRANTY.
    You are welcome to redistribute it under the terms of the
    GNU General Public License versions 2 or 3.
    For more information about these matters see
    https://www.gnu.org/licenses/.
    ```

### Installing New Packages

The `install.packages()` command in R will download new packages from the CRAN source directory and install them for your account. If you are running R from within a custom Anaconda environment, they will be specific to that environment. In either case, these packages will not be visible to other users.

### Checking Installed Packages

The command `installed.packages()` in R list details about all packages that are loaded and visible to current R session.

### Loading Packages

Packages are loaded into the current R environment through the `library()` function.

## Graphics

R is commonly used to produce high-quality graphics based on data. This capability is built-in and can be extended through the use of packages such as ggplot2. To produce graphics on the HPC systems, the easiest method is to output graphical displays to an appropriate filetype (pdf, jpeg, etc.). Then this file can be moved to your local machine using command line tools such as scp or rsync.

??? example "Example R Script for Graphics Output"

    ```R
    library(ggplot2)
    set.seed(8675309)
    numbers = rnorm(200, sd = 2)
    more.numbers = rnorm(100, mean = 10, sd = 2)

    df = data.frame(values = c(numbers, more.numbers))

    p = ggplot(df, aes(x = values, y = ..density..)) +
        geom_histogram(fill = "dodgerblue",
                       colour = "black",
                       alpha = .5,
                       binwidth = .5) +
        geom_density(size = 1.5) +
        labs(y = "Density", x = "Value",
             title = "Histogram Example")

    png(file = "histogram_example.png")
    print(p)
    dev.off()
    ```

## Parallel Programming in R

Programming in R on the HPC systems has two distinct advantages. First, running jobs on a remote system means you do not have to tie up your local machine. This can be particularly useful for jobs that take considerable time and resources to run. Secondly, the increased computational capabilities of the HPC system provide an opportunity to improve performance through parallel processing. R code, like many programming languages, is typically written and executed serially. This means that the added benefits of having multiple processing cores available are typically lost.

A major goal of the R community in recent years has been the development of specialized libraries and programming paradigms to better leverage modern HPC systems. The [CRAN Task View: High-Performance and Parallel Computing with R](https://cran.r-project.org/web/views/HighPerformanceComputing.html) contains a detailed list of packages that address various aspects of these problems. 

Notable examples are:

- Parallel
- Foreach
- Multicore
- Snow
- pbdR
- Rmpi

Each package includes in-depth documentation and examples for how to implement parallel processing in R code. Learning these packages does require a moderate amount of time, but for many large problems the improvements in computational efficiency dramatically outweighs the initial investment.

Most of these packages will have to be installed in a custom environment as many dependencies are incompatible with the version of openmpi installed in conda. 

??? note "Using the pbdR Project on Kestrel"

    The [pbdR project](http://r-pbd.org/) "enables high-level distributed data parallelism in R, so that it can easily utilize large HPC platforms with thousands of cores, making the R language scale to unparalleled heights." There are several packages within this project: pbdMPI for easy MPI work, pbdDMAT for distributed data matrices and associated functions, and pbdDEMO for a tutorial/vignette describing most of the project's details.
    
    The `pbdMPI` package provides the MPI interface, which requires Open MPI.  Note that the Open MPI module must be loaded prior to installing the package. For example, on Kestrel:
    
    ```
    $ module load openmpi/5.0.3-gcc
    $ R
    > install.packages("pbdMPI")
    ```

    The following script is a ranknode.R example using the pbdMPI package:

    ```R
    library(pbdMPI, quiet = TRUE)
    init()
    .comm.size <- comm.size()
    .comm.rank <- comm.rank()
    .hostname <- Sys.info()["nodename"]
    msg <- sprintf("I am %d of %d on %s.\n", .comm.rank, .comm.size, .hostname)
    comm.cat(msg, all.rank = TRUE, quiet = TRUE)
    comm.cat(msg, rank.print = sample(0:.comm.size, size = 1))
    comm.cat(msg, rank.print = sample(0:.comm.size, size = 1), quiet = TRUE)
    finalize()
    ```
    
    You could run this interactively from a compute node or by submitting it to the job scheduling using a shell script similar to the one given below. For example, you would submit this job to a [shared node on Kestrel](../../Systems/Kestrel/Running/index.md#shared-node-partition) via `sbatch ranknode.sh` from a login node provided you name the script appropriately: 

    ```bash
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=24
    #SBATCH --mem-per-cpu=1G
    #SBATCH --time=00:05:00
    #SBATCH --partition=shared
    #SBATCH --account=<your_allocation_id>

    module load mamba
    module load openmpi/5.0.3-gcc
    conda activate /path/to/renv

    INPUT_BASENAME=ranknode # JOB NAME - USER INPUT PARAMETER
    JOB_FILE=$INPUT_BASENAME.R
    OUT_FILE=$INPUT_BASENAME.Rout
    srun -n 48 Rscript $JOB_FILE > $OUT_FILE
    ```

    In either case (interactive or queue submission), the output produced from the ranknode.R script should look like this:

    ```
    I am 0 of 48 on x1004c0s2b0n0.
    I am 1 of 48 on x1004c0s2b0n0.
    I am 2 of 48 on x1004c0s2b0n0.
    ...
    I am 46 of 48 on x1004c0s2b0n1.
    I am 47 of 48 on x1004c0s2b0n1.
    I am 42 of 48 on x1004c0s2b0n1.
    I am 45 of 48 on x1004c0s2b0n1.
    ```

## Contacts

For questions on the R software environment itself or advanced R package questions, please contact [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov).

Additionally, NREL has an internal R Users Group that meets periodically to highlight interesting packages, problems, and share experiences related to R programming. For more details, contact [Daniel Inman](mailto:daniel.inman@nrel.gov). 

## References

- [Rmpi: Interface Wrapper to MPI (Message-Passing Interface)](http://cran.r-project.org/web/packages/Rmpi/index.html)
- [University of Western Ontario – Rmpi News](http://fisher.stats.uwo.ca/faculty/yu/Rmpi/)
- [State of the Art in Parallel Computing with R](http://www.jstatsoft.org/v31/i01)
