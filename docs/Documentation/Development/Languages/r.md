# Running R Statistical Computing Environment Software

*Learn how to run the R statistical computing environment software.*

## What Is R?

R is an open-source programming language designed for statistical computing and graphics. It is the current standard for the development of new statistical methodologies and enjoys a large user base.

For more information related to the R project, see the [R website](http://www.r-project.org/).

## Running R Interactively on Eagle

R is most commonly used via an interactive shell. To do this on Eagle, first request an interactive compute node ([see running interactive jobs on Eagle](../../Systems/Eagle/interactive_jobs.md)) using the srun command. Alternatively, R can be used through Europa running Jupyterhub. For more details, see [Jupyterhub](../Jupyter/jupyterhub.md).

Anaconda R is our actively supported distribution on Eagle. For more information about using R in the Anaconda framework, see [Using R language with Anaconda](https://docs.anaconda.com/free/anaconda/packages/using-r-language/). 

Once on a compute node, R environments can be accessed through the conda module (see [environment modules on the Eagle System](../../Systems/Eagle/modules.md) for general instructions on using modules).  To avoid possible conflicts, remove any Intel compiler modules before loading R. One way to do this is via the following: 

```
$ module purge
$ module load conda
```

To access the R interactive console using the default environment installed with conda, type R at the command line. You will be prompted with the familiar R console in your terminal window: 

??? note "R Terminal"

    ```
    $ R

    R version 4.0.5 (2021-03-31) -- "Shake and Throw"
    Copyright (C) 2021 The R Foundation for Statistical Computing
    Platform: x86_64-conda-linux-gnu (64-bit)

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

Another option for using R on Eagle is to submit them as part of job scripts to be run on non-interactive nodes. 

An example job script for running the hello_world.R example is below:

```bash
#! /bin/bash
#SBATCH --job-name=helloworld
#SBATCH --nodes=1
#SBATCH --time=60
#SBATCH --account=<your_allocation_id>
  
module purge
module load conda
Rscript hello_world.R
```
    
## Versions and Packages

R is a popular open-source language with an active development community. New versions of R are frequently released. Any version can be installed into a custom anaconda environment. Commands for using other versions is shown below: 

??? example "Custom Installation with Conda"

    ```
    $ conda search r-essentials
    Loading channels: done
    # Name                  Version           Build  Channel
    r-essentials                1.0        r3.2.1_0  pkgs/r
    r-essentials                1.0       r3.2.1_0a  pkgs/r
    r-essentials                1.1        r3.2.1_0  pkgs/r
    r-essentials                1.1       r3.2.1_0a  pkgs/r
    r-essentials                1.1        r3.2.2_0  pkgs/r
    r-essentials                1.1       r3.2.2_0a  pkgs/r
    r-essentials                1.1        r3.2.2_1  pkgs/r
    r-essentials                1.1       r3.2.2_1a  pkgs/r
    r-essentials                1.4               0  pkgs/r
    r-essentials              1.4.1        r3.3.1_0  pkgs/r
    r-essentials              1.4.2               0  pkgs/r
    r-essentials              1.4.2        r3.3.1_0  pkgs/r
    r-essentials              1.4.3        r3.3.1_0  pkgs/r
    r-essentials              1.5.0               0  pkgs/r
    r-essentials              1.5.1               0  pkgs/r
    r-essentials              1.5.2        r3.3.2_0  pkgs/r
    r-essentials              1.5.2        r3.4.1_0  pkgs/r
    r-essentials              1.6.0        r3.4.1_0  pkgs/r
    r-essentials              1.7.0  r342hf65ed6a_0  pkgs/r
    r-essentials              3.4.3        mro343_0  pkgs/r
    r-essentials              3.4.3          r343_0  pkgs/r
    r-essentials              3.5.0        mro350_0  pkgs/r
    r-essentials              3.5.0          r350_0  pkgs/r
    r-essentials              3.5.1        mro351_0  pkgs/r
    r-essentials              3.5.1          r351_0  pkgs/r
    $ conda create -n otherr r-essentials==3.5.1
    <Text>
    $ . activate otherr
    (otherr) $ R --version
    R version 3.5.1 (2018-07-02) -- "Feather Spray"
    Copyright (C) 2018 The R Foundation for Statistical Computing
    Platform: x86_64-pc-linux-gnu (64-bit)

    R is free software and comes with ABSOLUTELY NO WARRANTY.
    You are welcome to redistribute it under the terms of the
    GNU General Public License versions 2 or 3.
    For more information about these matters see
    http://www.gnu.org/licenses/.
    ```

### Installing New Packages

The `install.packages()` command in R will download new packages from the CRAN source directory and install them for your account. If you are running R from within a custom Anaconda environment, they will be specific to that environment. In either case, these packages will not be visible to other users.

### Checking Installed Packages

The command `installed.packages()` in R list details about all packages that are loaded and visible to current R session.

### Loading Packages

Packages are loaded into the current R environment through the `library()` function.

## Graphics

R is commonly used to produce high-quality graphics based on data. This capability is built-in and can be extended through the use of packages such as ggplot2. To produce graphics on Eagle, the easiest method is to output graphical displays to an appropriate filetype (pdf, jpeg, etc.). Then this file can be moved to your local machine using command line tools such as scp or rsync.

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

A major goal of the R community in recent years has been the development of specialized libraries and programming paradigms to better leverage modern HPC systems. The CRAN task view for High Performance Computing and Parallel Programming contains a detailed list of packages that address various aspects of these problems. For more information, see CRAN Task View: High-Performance and Parallel Computing with R. 

Notable examples are:

- Parallel
- Foreach
- Multicore
- Snow
- pbdR
- Rmpi

Each package includes in-depth documentation and examples for how to implement parallel processing in R code. Learning these packages does require a moderate amount of time, but for many large problems the improvements in computational efficiency dramatically outweighs the initial investment.

Most of these packages will have to be installed in a custom environment as many dependencies are incompatible with the version of openmpi installed in conda. 

??? note "Using the pbdR Project"

    The [pbdR project](http://r-pbd.org/) "enables high-level distributed data parallelism in R, so that it can easily utilize large HPC platforms with thousands of cores, making the R language scale to unparalleled heights." There are several packages within this project: pbdMPI for easy MPI work, pbdDMAT for distributed data matrices and associated functions, and pbdDEMO for a tutorial/vignette describing most of the project's details.

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
    
    You could run this interactively from a compute node or by submitting it to the job scheduling using a shell script similar to the one given below. For example, you would submit using sbatch ranknode.sh from a login node provided you name the script appropriately: 

    ```bash
    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=24
    #SBATCH --time=5
    #SBATCH --account=<your_allocation_id>

    module purge
    module load conda

    INPUT_BASENAME=ranknode # JOB NAME - USER INPUT PARAMETER
    JOB_FILE=$INPUT_BASENAME.R
    OUT_FILE=$INPUT_BASENAME.Rout
    mpirun -np 48 Rscript $JOB_FILE > $OUT_FILE
    ```

    In either case (interactive or queue submission), the output produced from the ranknode.R script should look like this:

    ```
    I am 0 of 48 on r1i5n1.
    I am 1 of 48 on r1i5n1.
    I am 2 of 48 on r1i5n1.
    ...
    I am 45 of 48 on r1i5n2.
    I am 46 of 48 on r1i5n2.
    I am 47 of 48 on r1i5n2.
    ```

## Contacts

For questions on statistics, the R software environment itself, or advanced R package questions, please contact [Lindy Williams](mailto:Lindy.Williams@nrel.gov).

Additionally, NREL has an internal R Users Group that meets periodically to highlight interesting packages, problems, and share experiences related to R programming. For more details, contact [Daniel Inman](mailto:daniel.inman@nrel.gov). 

## References

- [Rmpi: Interface Wrapper to MPI (Message-Passing Interface)](http://cran.r-project.org/web/packages/Rmpi/index.html)
- [University of Western Ontario – Rmpi News](http://fisher.stats.uwo.ca/faculty/yu/Rmpi/)
- [State of the Art in Parallel Computing with R](http://www.jstatsoft.org/v31/i01)
