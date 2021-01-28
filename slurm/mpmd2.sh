#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --out=%J.out
#SBATCH --error=%J.err

:<<++++

Author: Tim Kaiser

Here we look at launching Multi Program Multi Data run. 

We use a the --multi-prog option with srun.  This involves 
creating a config_file that lists the programs we are going 
to run along with the task ID.  See:

https://computing.llnl.gov/tutorials/linux_clusters/multi-prog.html

for a quick description of the format for the config_file.

We are using two applications c_ex02 and f_ex02.  We will be
launcing a single instance of c_ex02 and 7 copies of f_ex02.

It this case we also use hostlist which is a list of nodes
on which to run the various tasks of our simulation.

We will have c_ex02 run by itself on one node and the 7
instances of f_ex02 run on the second node.  In real life
you might want to do this, even with a single program if
task 0 will use a large amount of memory and the rest less.

The number of tasks running is fixed in this script at 8

++++


# load our version of MPI
module load mpt

#create our config_file 8 total tasks
echo "#" $SLURM_JOBID " config_file "> config_file
app1=./c_ex02
for n in 0 ; do
  echo $n $app1 >> config_file
done
app2=./f_ex02
for n in `seq 7`  ; do
  echo $n $app2 >> config_file
done

#create our hostlist file 8 total tasks
# counts should match what we have used in our config_file
counts="1 7"
# convert the counts string into an array
c_ray=(${counts})

# get a list of nodes
nodes=`scontrol show hostnames`
# convert the nodes string into an array
n_ray=(${nodes})
# iterate through our arrays and 
#print out "count" copies of the node
len=${#n_ray[@]}
rm -rf hostlist
for (( i=0; i<$len; i++ )); do
	for x in `seq ${c_ray[$i]}` ; do
		echo "${n_ray[$i]}" >> hostlist
    done
done

echo hostlist
cat hostlist
echo

echo config_file
cat config_file

echo
export SLURM_HOSTFILE=hostlist
srun -n 8 --multi-prog config_file | sort

:<<++++

Example output

el2:collect> sbatch -A hpcapps --partition=debug mpmd2.sh



el2:collect> cat 5371111.out
hostlist
r103u21
r103u23
r103u23
r103u23
r103u23
r103u23
r103u23
r103u23

config_file
# 5371111  config_file 
0 ./c_ex02
1 ./f_ex02
2 ./f_ex02
3 ./f_ex02
4 ./f_ex02
5 ./f_ex02
6 ./f_ex02
7 ./f_ex02

 getting            1
Hello from c process      :    0  Numprocs is    8 r103u21
Hello from fortran process:    1  Numprocs is    8 r103u23
Hello from fortran process:    2  Numprocs is    8 r103u23
Hello from fortran process:    3  Numprocs is    8 r103u23
Hello from fortran process:    4  Numprocs is    8 r103u23
Hello from fortran process:    5  Numprocs is    8 r103u23
Hello from fortran process:    6  Numprocs is    8 r103u23
Hello from fortran process:    7  Numprocs is    8 r103u23
 i=         200
el2:collect> 
++++

