#!/bin/bash
container_ids=$(sudo docker container ps -a -q)
IFS=''
read -ra container_id_list <<<"$container_ids"
container_id=${container_id_list[0]}
echo ${container_id}
#docker cp $container_id:workspace/jsrl-CORL/algorithms/finetune/losses_vf.png plots/
cp algorithms/finetune/losses_vf.png plots/
mv plots/losses_vf.png "plots/${container_id}_losses_vf.png"
#docker cp $container_id:workspace/jsrl-CORL/algorithms/finetune/pred_y.npy plots/
cp algorithms/finetune/pred_y.npy plots/
mv plots/pred_y.npy "plots/${container_id}_pred_y.npy"
#docker cp $container_id:workspace/jsrl-CORL/algorithms/finetune/true_y.npy plots/
cp algorithms/finetune/true_y.npy plots/
mv plots/true_y.npy "plots/${container_id}_true_y.npy"

