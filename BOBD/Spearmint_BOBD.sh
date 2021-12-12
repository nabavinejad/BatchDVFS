#!/bin/bash

for j in {1..30}
do
	rm config.json
	cat job${j}.json >> config.json

	echo "password_of_user" | sudo -S pkill nvidia-smi

	mongo < mongoDB_Commands.js
	sleep 5

	python /path/to/Spearmint/spearmint/main.py /path/to/python/files/ &
	PID=$!
	
	echo $PID > processID.txt

	wait $PID

	rm config.json

	sleep 5
done

exit 0
