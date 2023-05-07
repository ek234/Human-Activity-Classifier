#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mincpus=9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_svm_rbf_beyond.txt

LOGFILE="./log_svm_rbf_beyond.txt"

pids=()
for c in 0.1 1 10 100 1000; do
	for redness in red big; do
		python ../src/svm_beyond.py "$c" "$redness" "rbf" &
		pid="$!"
		echo "$pid: started svm at C=$c, size=$redness" &>> "$LOGFILE"
		pids+="$pid"
	done
done
echo "procs: $pids" &>> "$LOGFILE"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" &>> "$LOGFILE"
done
