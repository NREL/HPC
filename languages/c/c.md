1.  [HPC User Wiki](index.html)
2.  [NREL HPC User Community
    Wiki](NREL-HPC-User-Community-Wiki_15171667.html)
3.  [Tips and Tricks](Tips-and-Tricks_18593769.html)

 HPC User Wiki : C Programs 
===========================

Created by Southerland, Jennifer on 2018-01-23

Compiling and Running C Programs on the Peregrine System
========================================================

### Simple "Hello, world" Program

Follow these instructions to compile and run your first programs written
in C on Peregrine.

Copy this text to a file named hello\_world.c with your favorite text
editor (nano, vi, emacs, etc.).

    /* C Example */
    #include <stdio.h>

    int main (argc, argv)
       int argc;
       char *argv[];
    {
       printf( "Hello, world \n" );
       return 0;
    }

Compile this program using the Intel C compiler, icc:

    $ icc -o hello_world hello_world.c

The compiler will create a Linux executable file called hello\_world.

To run this program, use your editor to create a job script containing
the following text:

    #!/bin/bash -l
    #PBS -j oe
    #PBS -N job_hello_world
    #PBS -l walltime=01:00:00
    #PBS -l nodes=1

    # this ensures your job runs from the directory from which you run the qsub command
    cd $PBS_O_WORKDIR

    ./hello_world

Give the file a name like hello\_world.sh.

Submit the script using the qsub command and a valid project handle.
Because this is a very quick job, let's submit it to the "short" queue.

    $ qsub -q short -A CSC000 hello_world.sh

The output from this job is in a file with the name
job\_hello\_world.o&lt;JID&gt; where JID is the job id. You can see the
contents of that file using the "cat" command:

    [icarpent@login4 ~]$ cat job_hello_world.o1112070
    Warning: no access to tty (Bad file descriptor).
    Thus no job control in this shell.
    Hello, world

### Parallel "Hello, world" Program

For this example, we'll use MPI to write a parallel Hello World program
written in C. The Message Passing Interface (MPI)( is one way to create
programs that run on more than one processor.

Copy this text to a file named hello\_world.c with your favorite text
editor (vi, emacs, etc.)  If you are not familiar with text editors, try
"nano hello\_world.c"

    /* C Example */
    #include <stdio.h>
    #include <mpi.h> 


    int main (argc, argv)
       int argc;
       char *argv[];
    {
       int rank, size;
       MPI_Init (&argc, &argv); /* starts MPI */
       MPI_Comm_rank (MPI_COMM_WORLD, &rank);     /* get current process id */
       MPI_Comm_size (MPI_COMM_WORLD, &size);     /* get number of processes */
     
        printf( "Hello, world from process %d of %d\n", rank, size );
     
        MPI_Finalize();
       return 0;
    }

Compile this program using mpicc, which is a C compiler that knows about
MPI.

    mpicc hello_world.c -o hello_world

The compiler will create a linux executable file called hello\_world.

### Running your Parallel “Hello, world” program

Peregrine is a cluster: a collection of computers connected together
with a special network (InfiniBand in this case) and software (Torque)
that allows a single program to run across multiple physical computers.
Create a submit script which contains options for torque, including the
number of nodes you want your parallel program to run on and
instructions for how to start the parallel program. For example, create
a file named **hello.qsub** with the contents below.

    #!/bin/bash -l
    #PBS -j oe
    #PBS -N job_hello_world
    #PBS -l walltime=01:00:00
    #PBS -l nodes=2                 # this asks for 2 nodes 

    # this ensures your job runs from the directory from which you run the qsub command
    cd $PBS_O_WORKDIR
    set -x

    mpirun -np 32 ./hello_world >& my_results.out

Because this is an MPI program, we execute it using the mpirun command
and we tell it how many processors to run it on with the -np option. In
this case we have told the system to run the hello\_world program on 32
processors.  The nodes assigned to your job will have either 16 or 24
processors so to use 32 we have asked for 2 nodes. The system will use
some processors on the first node and some on the second node. Each
processor will run the program and print a line of output.

Once you've saved this file, next create your run directory in /scratch
using the mkdir command and copy your program and job script to it.

    $ mkdir /scratch/$USER/hello
    $ cp hello_world /scratch/$USER/hello
    $ cp hello.qsub /scratch/$USER/hello

Go to this directory using the cd command:

    $ cd /scratch/$USER/hello

Now, submit your job script using the qsub command with a valid project
handle:

    $ qsub hello.qsub -A project-handle

The job will run, then you should see an output file named
"my\_results.out" with one line of output from each MPI process.

    [kregimba@login1 hello]$ cat my_results.out
    Hello world from process 14 of 32
    Hello world from process 30 of 32
    ...
    Hello world from process 26 of 32
    [kregimba@login1 hello]$

Document generated by Confluence on 2019-04-03 10:15

[Atlassian](http://www.atlassian.com/)
