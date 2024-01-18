WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07
RUN_FILE = "jsrl-CORL/algorithms/finetune/iql.py"

run:
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	--gpus all \
	-it \
	jsrl-corl python $(RUN_FILE) --env "antmaze-umaze-v2" --n_episodes 100 --seed 3 --discount 0.99 --iql_tau 0.9 --beta 10 --normalize_reward True --normalize True --batch_size 256 --buffer_size 10000000 --device cuda --eval_freq 5000 --iql_deterministic False 

build:
	sudo docker build \
	-t jsrl-corl \
	.
