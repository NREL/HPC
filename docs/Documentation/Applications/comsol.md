---
title: Comsol
---

# Using COMSOL Software 
*COMSOL Multiphysics is a versatile finite element analysis and simulation package. The COMSOL Desktop GUI (graphical user interface) offers an environment for building and solving models. It provides script-based modeling capabilities.*

Currently, we host three floating network licenses and a number of additional modules. Issue the command `lmstat.comsol` to see current license status and COMSOL modules available.

# Building a COMSOL Model
Extensive documentation is available in the menu: Help > Documentation and in Help > Dynamic Help. For beginners, it is highly recommended to follow the steps in Introduction to COMSOL Multiphysics found in Help > Documentation.

For instructional videos, see the COMSOL (website)[https://www.comsol.com].

Running COMSOL Interactively
License status, including how many licenses are presently checked out, can be viewed by invoking the following command:

```
[user@el3 ~]$ lmstat.comsol
```

COMSOL can be used by starting the COMSOL GUI that allows one to build models, run the COMSOL computational engine, and analyze results. With the COMSOL engine running on Eagle, the input to and output from the GUI must be available on a remote machine. The remote machine must be able to send/receive information for X Windows.

From the Terminal application on a FastX desktop, the following should bring up the COMSOL interface.

```
[user@ed3 ~]$ module purge
[user@ed3 ~]$ module load comsol/5.4
[user@ed3 ~]$ vglrun comsol
```

From an X-enabled shell on a compute node, replace the last command with

```
[user@r1i7n24 ~]$ comsol -3drend sw
```

Running a COMSOL Job in Batch Mode
You can save your model built in GUI mode into a file such as myinputfile.mph. Once that's available, the following job script shows how to run a single process multithreaded job in batch mode:

```bash
#!/bin/bash
#SBATCH --job-name=comsol-batch-1proc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --account=<your-allocation-id>
#SBATCH --output=comsol-%j.out
#SBATCH --error=comsol-%j.err

# This helps ensure your job runs from the directory
