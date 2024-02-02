#!/bin/bash
done=$(sudo docker container ps -q)

while [ -n "$done" ]
do
	echo "$done"
	done=$(sudo docker container ps -q)
done
