#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_lr.txt

LOGFILE="./log_lr.txt"

which python

pids=()
for c in 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10; do
	for redness in red big; do
		python ../src/logistic_regression.py "$c" "$redness" &
		pid="$!"
		echo "$pid: started lr at C=$c, size=$redness" >> "$LOGFILE"
		pids+="$pid"
	done
done
echo "procs: $pids"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" >> "$LOGFILE"
done
