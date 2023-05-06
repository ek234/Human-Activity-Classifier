#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_svm_poly.txt

LOGFILE="./log_svm_poly.txt"

pids=()
for c in 0.1 1 10 100 1000; do
	for redness in red big; do
		python ../src/svm.py "$c" "$redness" "poly" &
		pid="$!"
		echo "$pid: started svm at C=$c, size=$redness" >> "$LOGFILE"
		pids+="$pid"
	done
done
echo "procs: $pids"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" >> "$LOGFILE"
done