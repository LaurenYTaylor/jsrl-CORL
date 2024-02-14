#!/bin/bash
done=$(sudo docker container ps -q)

while [ -n "$done" ]
do
	done=$(sudo docker container ps -q)
done

make -f Makefile_MultiRun_WSL build_and_run
