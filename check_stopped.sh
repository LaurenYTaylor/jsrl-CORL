#!/bin/bash
done=$(sudo docker container ps | grep jsrl-corl)

while [ -n "$done" ]
do
	done=$(sudo docker container ps | grep jsrl-corl)
	echo "$done"
done
