#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
doits="""<!-- 000 -->,script
<!-- 001 -->,cuda/cray/doit
<!-- 002 -->,cuda/gccalso/doit
<!-- 003 -->,cuda/nvidia/doit
<!-- 004 -->,mpi/normal/cray/doit
<!-- 005 -->,mpi/normal/intel+abi/doit
<!-- 006 -->,mpi/normal/nvidia/nrelopenmpi/doit
<!-- 007 -->,mpi/normal/nvidia/nvidiaopenmpi/doit
<!-- 008 -->,mpi/withcuda/cray/doit
<!-- 009 -->,mpi/withcuda/nvidia/nrelopenmpi/doit
<!-- 010 -->,mpi/withcuda/nvidia/nvidiaopenmpi/doit
<!-- 011 -->,mpi/cudaaware/doit
<!-- 012 -->,openacc/cray/doit
<!-- 013 -->,openacc/nvidia/doit
<!-- 014 -->,mpi/openacc/cray/doit
<!-- 015 -->,mpi/openacc/nvidia/nrelopenmpi/doit
<!-- 016 -->,mpi/openacc/nvidia/nvidiaopenmpi/doit
<!-- 017 -->,cudalib/factor/doit
<!-- 018 -->,cudalib/fft/doit"""
doits=doits.split("\n")
todo={}
for t in doits:
    case,name=t.split(",")
    todo[case]=name
okeys=list(todo.keys())
okeys.sort()


# In[ ]:


f=open("about.md","r")
old=f.read()
old=old.split("\n")


# In[ ]:

fordoc=False
if len(sys.argv)> 1:
	fordoc=True
ocount=0
looking=okeys[ocount]
insection=False
for l in old:    
    if(l.find(okeys[ocount]) > -1):
        #print(todo[looking])
        print(l)
        newfile=open(todo[looking],"r")
        newdat=newfile.read()
        if fordoc :
            print('??? example "',todo[looking],'"')
            print("\t```bash")
            newdat=newdat.split("\n")
            for tline in newdat:
                print("\t",tline)
            print("\t```")        
        else:
            print("\n```")
            newdat=newdat.split("\n")
            for tline in newdat:
                print(tline)
            print("```")
        newfile.close        
        insection=True
        tick=-2
        if(ocount<len(okeys)-1):
            ocount=ocount+1
            looking=okeys[ocount]
    if (not(insection)):
        print(l)
    if(insection):
        if(l.find("```") > -1):
            tick=tick+1
            if(tick==0): insection=False
        
