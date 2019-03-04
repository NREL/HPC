cd /scratch/`whoami`
mkdir -p cpu-rollcall
cd cpu-rollcall

echo "Unordered rollcall from each CPU:"
srun --cpus-per-task=1 --ntasks=36 bash <<< 'CPU_ID=$(cat /proc/self/stat); echo $CPU_ID | gawk "{print \$39}"' > cpu.ranks

echo `tr '\n' ' ' < cpu.ranks` # print unordered ranks, showing they ran parallely

# Now use cores to merge-sort the rank prints
# NOTE: performance wise, this sort is terrible, this is only to make it obvious what's happening
echo
echo "Now let's use different cores as independent workers to do a merge sort of this printout:"
IFS=$'\n'
tr '\n' '@' < cpu.ranks | sed -E 's/([0-9]+)@([0-9]+)@/\1 \2\n/g' > cpu.sort

cat cpu.sort
echo

while (( `wc -l cpu.sort | awk '{print $1}'` > 1)); do
	
	lines=(`cat cpu.sort`)
	> cpu.sort
	for ((i=0; i<${#lines[@]};i+=2)); do
		printf "${lines[i]} ${lines[i+1]}\n" >> cpu.sort
	done
	
	> cpu.newsort
	batches=`cat cpu.sort`
	for batch in $batches; do
		srun --cpu_bind=cores --cpus-per-task=1 --ntasks=1 bash <<< "
			tr ' ' '\n' <<< \"$batch\" | sort -n | tr '\n' ' ' | xargs echo 
		" >> cpu.newsort &
	done
	wait
	mv cpu.newsort cpu.sort
	cat cpu.sort
	echo
done

echo "Done!"

cd ..
rm -rf ./cpu-rollcall
