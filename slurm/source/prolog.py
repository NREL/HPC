#!/usr/bin/env python3
#
# A simple prolog program for the srun command.
# This is run by adding the option
#  --task-prolog=./prolog.py 
# to your srun command.
# All this does is print information about the
# current run in a file with a name that is 
# a concatenation of the node name and pid of
# the process being started by srun. This program
# will be run before your real application starts
# See the srun man page for more information.
#
# NOTE:  This uses python 3 so you must load
# a module that brings in python 3 because it
# is not on the compute node by default. For 
# example:
#  module load conda
#
from subprocess import Popen, PIPE
from re import split
from sys import stdout
import os
import socket
import time

# Function to do a ps
def get_proc_list():
    ''' Retrieves a list [] of Proc objects representing the active
    process list list '''
    proc_list = []
    sub_proc = Popen(['ps', 'aux'], shell=False, stdout=PIPE)
    #Discard the first line (ps aux header)
    sub_proc.stdout.readline()
    for line in sub_proc.stdout:
        #The separator for splitting is 'variable number of spaces'
        proc_list.append(str(line))
    return proc_list

# Get time, hostname, and pid
now=time.time()
tstr=time.asctime(time.localtime(now))
host=socket.gethostname()
pid=os.getpid()

# Get the pid of the task being launched by srun
try:
     spid=os.environ['SLURM_TASK_PID']
except:
    spid=1
spid=int(spid)

# Open a file based on node name and pid of process to be launched by srun
name="{}.{:06d}".format(host,spid)
f=open(name,"w")

# Write out host, time, my pid, pid of process launched by srun
f.write(host+" "+str(now)+" "+tstr+" "+str(pid)+" "+str(spid)+"\n")

# Write out the SLURM variables
for e in os.environ:
	if str(e).find("SLURM") > -1 :
		line="{}{}{}".format(str(e),"\n\t",str(os.environ[e]))+"\n"
		f.write(line)
# Do a ps
plist=get_proc_list()
# Just print the process launced by srun.  Note that this is not
# the final program to be launched.  It is replaced via fork?
for p in plist:
	p=p.replace("\\n","")
	if p.find(" "+str(spid)+" ") > -1:
		f.write(p+"\n")
f.close()
#time.sleep(10)
