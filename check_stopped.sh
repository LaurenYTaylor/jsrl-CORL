#!/bin/bash
done=$(sudo docker container ps -q)
count=0
while [ ! -n $done ]
do
    echo $done
    done=$(sudo docker container ps -q)
done