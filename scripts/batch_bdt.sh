#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_bdt.txt

LOGFILE="./log_bdt.txt"
which python
pids=()
for d in 1 4 7 10; do
	for num_trees in 50 100 250 500; do
		for redness in red big; do
			python ../src/boosted_dt.py "$d" "$redness" "$num_trees" &
			pid="$!"
			echo "$pid: started bdt at Depth=$d, num_trees=$num_trees, size=$redness" >> "$LOGFILE"
			pids+="$pid"
		done
	done
done
echo "procs: $pids"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" >> "$LOGFILE"
done
