---
search:
  exclude: true
---

# Approaches to Debugging at Scale

On an HPC system, occasionally there is the need to debug programs at relatively large scale, on a larger number of nodes than what is available via the short or debug queues. Because many of jobs run for several days, it may take a long time to acquire a large number of nodes.

To debug applications that use many nodes, there are three possible approaches.

??? abstract "Approach 1: Run an Interactive Job"
    Submit an [interactive job](../../../Slurm/interactive_jobs.md) asking for the number of tasks you will need. For example:

    ```srun -n 3600 -t 1-00 -A <handle> --pty $SHELL```
    This asks for 3600 cores for 1 day. When the nodes are available for your job, you "land" in an interactive session (shell) on one of the compute nodes. From there you may run scripts, execute parallel programs across any of the nodes, or use an interactive debugger such as ARM DDT.

    When you are done working, exit the interactive session.

    Rarely will a request of this size and duration start right away, so running it within a ***screen session*** allows you to wait for your session to start without needing to stay connected to the HPC system.  With this method, users must periodically check whether their session has started by reconnecting to their screen session.

    Using screen sessions:

    1. On a login node, type "screen"

    1. Check to see whether your environment is correct within the screen session. If needed, purge modules and reload:
    ```
    [user@login2 ~]$ screen

    [user@login2 ~]$ module purge
    [user@login2 ~]$ module load PrgEnv-intel
    ```
    1. Request an interactive job:

    ```$ srun -n 3600 -t 1-00 -A <handle> --pty $SHELL```
    When you want to disconnect from the session, type ```control-a``` then ```control-d```. The interactive job continues to run on the HPC system.

    Later, to continue working in the interactive job session, reconnect to this screen session. To reconnect, if you have logged out of the system, first log in to the same login node. Then type ```screen -r``` to reattach to the screen session. If your interactive job has started, you will land on the compute node that you were given by the system.

    When you are done with your work, type ```exit``` to end the interactive job, and then type ```exit``` again to end the screen session.

??? abstract "Approach 2: Request a Reservation"
    A more convenient approach may be to request a reservation for the number of nodes you need.  A reservation may be shared by multiple users, and it starts and ends at specific times.  
    
    To request a reservation for a debugging session, please [contact us](mailto://hpc-help@nrel.gov) and include:

        * Project handle 
        * Number of nodes 
        * Time of the request
        
    When the work is complete, please inform the HPC Operations team, so the reservation can be released. The project allocation will be charged for the reserved time, up until the reservation is released, whether that time is used or not.

    When your reserved time starts you may run either interactive jobs or regular batch jobs on the nodes in the reservation.

??? abstract "Approach 3: Offline Debugging"
    It might be difficult to debug a large parallel job on an HPC system interactively. An alternative is to debug the problem by submitting a job for offline debugging. 

    The problem should be scaled down such that it can easily get access to an interactive queue (around 2 nodes). Create an interactive session and open the ARM DDT debugger(GUI). Run the program and set evaluations, tracepoints, watchpoints etc in the DDT session. Save the session file. 

    You can then submit a larger job with ARM DDT in offline mode pointing to the session file created in the previous step. At the end of the run, you can view the generated debugging report in html or text mode.
