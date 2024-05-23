WANDB_API_KEY=b3fb3695850f3bdebfee750ead3ae8230c14ea07
DF=Dockerfile
RUN_FILE="jsrl-CORL/algorithms/finetune/ray_trainer.py"
CPUS=5

run:
	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--rm \
	jsrl-corl python $(RUN_FILE)

build:
	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

build_and_run_lunar:
	yes | sudo docker container prune

	sudo docker build \
	-f $(DF) \
	-t jsrl-corl \
	.

	sudo docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	$(DETACH) \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/jsrl-CORL" \
	jsrl-corl python $(RUN_FILE) --env LunarLander-v2 --variance_learn_frac 0.1 --guide_heuristic_fn lunar_lander --offline_iterations 0 --env_config '{"continuous": True}' --eval_freq 500 --beta 10 --iql_tau 0.9 --horizon_fn variance --name IQL-test --device cpu --online_iterations 1 ;


build_and_run_antmaze:
	yes | docker container prune

	docker build \
	-f $(DF) \
	-t jsrl-corl \
	.
	docker run \
	-e WANDB_API_KEY=$(WANDB_API_KEY) \
	-it \
	--shm-size=10.24gb \
	$(DETACH) \
	--cpus $(CPUS) \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/jsrl-CORL" \
	jsrl-corl python $(RUN_FILE) --horizon_fn time_step --checkpoints_path checkpoints --env antmaze-large-play-v2 --normalize True --tolerance 0.05 --normalize_reward True --iql_deterministic False --beta 10 --iql_tau 0.9 --eval_freq 10000 --n_episodes 20 --n_curriculum_stages 5 --no_agent_types True --new_online_buffer True --pretrained_policy_path jsrl-CORL/algorithms/finetune/checkpoints/IQL-antmaze-large-play-v2-offline/checkpoint_999999.pt --offline_iterations 0 --online_iterations 1000000 --device cpu ;

run_variance_learner:
	sudo docker run \
	-it \
	-v ./algorithms/finetune/checkpoints:/workspace/checkpoints \
	-v ./algorithms/finetune/var_functions:/workspace/jsrl-CORL/algorithms/finetune/var_functions \
	-v ./algorithms/finetune/wandb:/workspace/wandb \
	-v ".:/workspace/jsrl-CORL" \
	jsrl-corl python jsrl-CORL/algorithms/finetune/variance_learner.py
