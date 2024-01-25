#!/bin/bash
done=$(sudo docker container ps -q)
count=0
while [ $done ] && [ $count -le 10 ] 
do
    echo $done
    count=$((count+1))
done