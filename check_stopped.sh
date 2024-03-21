#!/bin/bash
done=$(sudo docker container ps | grep jsrl-corl)

while [ -n "$done" ]
do
	echo "$done"
	done=$(sudo docker container ps | grep jsrl-corl)
done