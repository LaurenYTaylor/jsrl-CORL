WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07
RUN_FILE = "jsrl-CORL/algorithms/offline/iql.py "

run:
	sudo docker run 
	-e WANDB_API_KEY=$(WANDB_API_KEY)\
	--gpus all \
	-it \
	jsrl-corl python $(RUN_FILE)

build:
	sudo docker build\
	-t jsrl-corl\
	.