#!/usr/bin/python
import os
import sys
import random

"""
Set up a file, in_list, that has input for a collection of array jobs
then run them using the script uselist.sh.  Input to this script is the 
account to use to run the jobs.

Example Run:

el2:array> ./doarray.py hpcapps

Example Output:

COMMAND:
sbatch -A hpcapps --array=1-24 uselist.sh
Submitted batch job 5400715

###################
We the use squeue to see what is in the queue.

el2:array> squeue -u tkaiser2
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
    5400715_[1-24]     short array_jo tkaiser2 PD       0:00      1 (Priority) 
el2:array> 

The script uselist.sh pulls a single line from "in_list" for each instance
and uses if for input for the application it is going to run.  The 
application is a matrix inversion program.  n1-n5 are used to set up
the matricies.  

"""

size=24
# make list of inputs
l=open("in_list","w")
for x in range(0,size):
	n1=int(random.random()*99)+2
	n2=int(random.random()*99)+2
	n3=int(random.random()*99)+2
	n4=int(random.random()*99)+2
	n5=400
	l.write("%d %d %d %d %d\n" % (n1,n2,n3,n4,n5))
l.close()
print("created in_list")
if len(sys.argv) < 2 :
	print("\nNormal USAGE:")
	print(sys.argv[0]+" account")
	sys.exit()
account=sys.argv[1]
command="sbatch -J slurm_test -A ACCOUNT --array=1-COUNT uselist.sh"
command=command.replace("ACCOUNT",account)
command=command.replace("COUNT",str(size))
print("COMMAND:")
print(command)
doit=os.popen(command,"r")
output=doit.readlines()
for o in output:
	print(o)





	
