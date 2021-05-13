#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:2
#SBATCH --out=%J.out
#SBATCH --error=%J.err


:<<++++

Author: Tim Kaiser

Some workflows combine applications that use GPUs and those
that only use CPUs.  It is possible to run GPU applications on
GPU nodes using a subset of the CPUs and simultainiously use
some of the remaining CPUs to run an additional application.

The difficulty here is if you ask slurm to give you GPU access
it will normally expect every application in your workflow to
use them.  Thus it will wait until the GPUs are free to run
the CPU only applicaiton.  The workaround for this situation 
is to unset the environmental variables while the GPUs are
running. 

Submission line:

sbatch -p debug -A hpcapps gpucpu.sh

++++

  
# load our version of MPI
module purge   
# needed for threaded apps built with Intel compilers
module load comp-intel
module load mpt
module load cuda/10.2.89
   
# Make a directory for our run and go there.
mkdir $SLURM_JOB_ID   
cat $0 > $SLURM_JOB_ID/script   
cd $SLURM_JOB_ID   
   
# run our MPI/GPU application but put it in the background
srun  -n 2 --gpus=2 -o mpigpu.out ../mpigpu  &   
sleep 10   
date   

# unset these to allow slurm to schedule the CPUs   
unset SLURM_JOB_GPUS   
unset GPU_DEVICE_ORDINAL   
unset SLURM_GPUS_PER_NODE   
   
# run our hybrid MPI/OpenMP  application
export OMP_NUM_THREADS=6   
srun --gpus=0 -o six.out -n 2 ../phostone -F -t 30    
   
# Since we put the mpigpu application in the background we need 
# to wait for it to finish before we hit the bottom of our slurm
# script or slurm could exit before it finishes.
wait   

:<<++++

Example output

Here instead of looking at the actual program output we ssh to the
node and run "ps" to show that both applications are running.  We
also run nvidia-smi to show the GPUs are being used.

+-----------------------------------------------------------------------------+
el2:collect> ssh r104u33  ps -U $USER -L -o pid,lwp,psr,comm,pcpu | grep -v COMMAND | sort -k3n | egrep "phostone|mpigpu"
28907 28930   1 mpigpu           0.0
28968 29061   1 phostone         100
28907 28938   2 mpigpu           0.0
28968 28997   2 phostone         0.0
28968 29060   2 phostone         100
28907 28907   3 mpigpu          67.8
28968 29059   4 phostone         100
28968 29058   5 phostone         100
28968 29057   6 phostone         100
28907 28939   9 mpigpu           0.0
28968 28968  17 phostone         100
28969 29066  19 phostone         100
28969 29065  20 phostone         100
28969 28996  22 phostone         0.0
28969 29064  22 phostone         100
28969 29063  23 phostone         100
28908 28908  24 mpigpu          67.8
28908 28937  25 mpigpu           0.0
28969 29062  25 phostone         100
28908 28929  26 mpigpu           0.0
28908 28940  27 mpigpu           0.0
28969 28969  35 phostone         100
el2:collect> ssh r104u33 nvidia-smi 
Mon Dec 21 10:51:35 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:37:00.0 Off |                    0 |
| N/A   41C    P0    42W / 250W |    321MiB / 16160MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00000000:86:00.0 Off |                    0 |
| N/A   43C    P0    45W / 250W |    321MiB / 16160MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     28907      C   /home/tkaiser2/collect/5399183/../mpigpu     309MiB |
|    1     28908      C   /home/tkaiser2/collect/5399183/../mpigpu     309MiB |
+-----------------------------------------------------------------------------+
el2:collect> 

++++
