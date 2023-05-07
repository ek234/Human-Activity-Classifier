#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_rf.txt

LOGFILE="./log_rf.txt"

pids=()
python ../src/random_forest.py &
pid="$!"
echo "$pid: started rf" >> "$LOGFILE"
pids+="$pid"

python ../src/random_forest_reduced.py &
pid="$!"
echo "$pid: started rf reduced" >> "$LOGFILE"
pids+="$pid"

        
echo "procs: $pids"

for job in $pids; do
	wait $job
	echo "$job: ended with status $?" >> "$LOGFILE"
done
