#!/bin/sh

for c in 0.1 1 10 100 1000; do
	for redness in red big; do
		python svm.py "$c" "$redness" &
	done
done

FAIL=0
for job in `jobs -p`; do
	echo $job
	wait $job || let "FAIL+=1"
done

if [ "$FAIL" == "0" ]; then
	echo "YAY!"
else
	echo "FAIL! ($FAIL)"
fi
