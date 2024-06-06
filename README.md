# JSRL (Jump-Start Reinforcement Learning)

This is an implementation of Jump-Start Reinforcement Learning (JSRL) (https://arxiv.org/abs/2204.02372) on top of IQL (more algorithms planned) from the Clean Offline Reinforcement Learning library. This is a fork of that library. I am not an author of JSRL.

The best way to run this is using the functions in the Makefile, which runs a Dockerfile. D4RL is a bit finicky so this is recommended.

For example to run with antmaze:
```
make -f build_and_run_antmaze
```

This calls RUN_FILE ray_trainer.py, you can use a different ray trainer with more seeds (or just increase the number of seeds in ray_trainer.py) if you have more resources.
