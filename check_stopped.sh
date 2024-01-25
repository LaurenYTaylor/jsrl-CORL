#!/bin/bash
done=$(sudo docker container ps -q)
count=0
while [ $done ]
do
    echo $done" "
done