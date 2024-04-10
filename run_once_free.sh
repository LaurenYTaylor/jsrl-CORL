#!/bin/bash
done=$(sudo docker container ps | grep jsrl-corl)

while [ -n "$done" ]
do
	done=$(sudo docker container ps | grep jsrl-corl)
done

make -f Makefile_MultiRun build_and_run
make -f Makefile_MultiRun_GPU1 build_and_run
