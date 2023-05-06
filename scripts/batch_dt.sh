#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_dt.txt

LOGFILE="./log_dt.txt"
which python
pids=()
for d in 2 5 10 20 50 100; do
	for redness in red big; do
		python ../src/decision_tree.py "$d" "$redness" &
		pid="$!"
		echo "$pid: started dt at Depth=$d, size=$redness" >> "$LOGFILE"
		pids+="$pid"
	done
done
echo "procs: $pids"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" >> "$LOGFILE"
done
