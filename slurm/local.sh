#!/bin/bash
#SBATCH --job-name="local"
#SBATCH --nodes=2
#SBATCH --time=00:02:00
#SBATCH --partition=debug


:<<++++

Author: Tim Kaiser

Script to show usage of local storage on Eagle.  

Creates directories on local storage on each node and
puts a file with random data there.  Shows the beginning
content of the files using "od"

Uses the slurm command sbcast to copy a file to local storage.  

Creates a directory in shared storage and copies the files from
local storage to the shared file system.

Deletes the local files in tmp on each node.

Note:
       2>&1 | grep -v "X11 forwarding"
    
    surpresses a warning from ssh.

Usage:
    sbatch -A hpcapps local.sh 

++++

cd $SLURM_SUBMIT_DIR

# Create a short JOBID base on the  one provided by the scheduler
JOBID=`echo $SLURM_JOBID`

# Make a directory for this run based on the JOBID and go there
# This should be on a directory shared by all nodes, that is 
# /scratch/$USER, thus we call it $SHARED

mkdir -p $JOBID
cd $JOBID
export SHARED=`pwd`

# On Eagle local disk is at /tmp/scratch
export JOBTMP=/tmp/scratch

sleep 10

#get a list of nodes...
export nlist=`scontrol show hostnames`
echo Running on: $nlist

# On the compute nodes we are going to create a local directory
# on tmp, again we use the JOBID for the name.

export MY_TMP=$JOBID

echo putting data

# For each node...
for i in $nlist
do 
# Create my temporary directory in /scratch on each node
  echo $i
  ssh -X $i "mkdir -p $JOBTMP/$MY_TMP" 2>&1 | grep -v "X11 forwarding"
# Create a file on each node with random data
  ssh -X  $i "head -c 1048576 </dev/urandom > $JOBTMP/$MY_TMP/afile" 2>&1 | grep -v "X11 forwarding"
# Get the first line of the file in octal
  ssh -X  $i "od $JOBTMP/$MY_TMP/afile | head -1" 2>&1 | grep -v "X11 forwarding"
  
done

# You can use the slurm command sbcast to put a file on each node
# here we just make a copy of your script and copy it to all nodes
cat $0 > script.$JOBID
sbcast script.$JOBID $JOBTMP/$MY_TMP/script.$JOBID

echo 
echo getting data
# For each node...
for i in $nlist
do 
  echo $i
# Copy files from our local space on each node back to
# my working directory creating a subdirectory for each node.
# This sould be two files, our random data and a copy of our script.
  mkdir -p $SHARED/$i
  scp  -r $i:$JOBTMP/$MY_TMP/* $SHARED/$i 2>&1 | grep -v "X11 forwarding"

# Get the first line of the file in octal
  od $SHARED/$i/afile | head -1 
  
# Remove the temporary directory
  ssh -X  $i "rm -r $JOBTMP/$MY_TMP" 2>&1 | grep -v "X11 forwarding"
done

echo 
echo get listing
ls -ltR .

# Copy our output file to our new directory
cp $SLURM_SUBMIT_DIR/slurm-$JOBID.out .



:<<++++

Example output:


el3:tkaiser2> sbatch -A hpcapps local.sh 
Submitted batch job 5414451
el3:tkaiser2> 
el3:tkaiser2> 
el3:tkaiser2> 
el3:tkaiser2> ls 5414451
r105u33  r105u37  script.5414451  slurm-5414451.out
el3:tkaiser2> 
el3:tkaiser2> 
el3:tkaiser2> cat slurm-5414451.out
Running on: r105u33 r105u37
putting data
r105u33
0000000 054330 116353 012303 063425 164175 011147 172011 032132
r105u37
0000000 023336 021710 016745 027242 041354 055422 126172 072471

getting data
r105u33
0000000 054330 116353 012303 063425 164175 011147 172011 032132
r105u37
0000000 023336 021710 016745 027242 041354 055422 126172 072471

get listing
.:
total 9
drwxrwx--- 2 tkaiser2 tkaiser2 4096 Dec 22 11:56 r105u37
drwxrwx--- 2 tkaiser2 tkaiser2 4096 Dec 22 11:56 r105u33
-rw-rw---- 1 tkaiser2 tkaiser2 2154 Dec 22 11:56 script.5414451

./r105u37:
total 1
-rw-rw---- 1 tkaiser2 tkaiser2 1048576 Dec 22 11:56 afile
-rw-rw---- 1 tkaiser2 tkaiser2    2154 Dec 22 11:56 script.5414451

./r105u33:
total 1
-rw-rw---- 1 tkaiser2 tkaiser2 1048576 Dec 22 11:56 afile
-rw-rw---- 1 tkaiser2 tkaiser2    2154 Dec 22 11:56 script.5414451
el3:tkaiser2> 

++++