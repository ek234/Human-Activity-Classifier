#!/bin/zsh

LOGFILE="./log.txt"

which python

pids=()
for c in 0.1 1 10 100 1000; do
	for redness in red big; do
		python ../src/svm.py "$c" "$redness" &
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
