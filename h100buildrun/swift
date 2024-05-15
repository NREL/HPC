#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --output=output-%j.out
#SBATCH --error=infor-%j.out


#make sure we are not using old modules
module unuse /nopt/nrel/apps/210928a/modules
module unuse /nopt/nrel/apps/210928a/level02/modules/lmod/linux-rocky8-x86_64/openmpi/4.1.1-mkxx6h3/Core
module unuse /nopt/nrel/apps/210928a/level02/modules/lmod/linux-rocky8-x86_64/Core
module unuse /nopt/nrel/apps/210928a/level01/modules/lmod/linux-rocky8-x86_64/Core
module unuse /nopt/nrel/apps/210928a/level00/modules/lmod/linux-rocky8-x86_64/gcc/9.4.0


# a simple timer
dt ()
{
    now=`date +"%s.%N"`;
    if (( $# > 0 )); then
        rtn=$(printf "%0.3f" `echo $now - $1 | bc`);
    else
        rtn=$(printf "%0.3f" `echo $now`);
    fi;
    echo $rtn
}

startdir=`pwd`
doswifts=`find . -name doswift`
for x in $doswifts ; do
	echo found example in `dirname $x`
done
t1=`dt`
for x in $doswifts ; do
 dir=`dirname $x`
 echo ++++++++ $dir >&2 
 echo ++++++++
 echo $dir
 cd $dir
 tbegin=`dt`
 . doswift
 echo Runtime `dt $tbegin` $dir `dt $t1` total
 cd $startdir
done
echo FINISHED `dt $t1`

