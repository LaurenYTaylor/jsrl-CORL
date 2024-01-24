WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07

RUN_FILE="jsrl-CORL/algorithms/finetune/iql.py"
NUM_GPUS=--gpus all
DETACH=--detach

run:
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--gpus all \
	--rm \
	jsrl-corl python $(RUN_FILE)

build:
	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

build_and_run:
	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=5.04gb \
	$(NUM_GPUS) \
	$(DETACH) \
	--cpus 6 \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	jsrl-corl python $(RUN_FILE) --checkpoints_path checkpoints

build_and_run_finetune_test:
	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--gpus all \
	--rm \
	jsrl-corl python $(RUN_FILE) --group IQL-D4RL-finetune_test --offline_iterations 5

	echo "running $(RUN_FILE)"
