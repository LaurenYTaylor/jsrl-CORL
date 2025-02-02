import pyrallis
import os
from jsrl_w_iql import JsrlTrainConfig
from eval_w_render import evaluate
import time
import wandb

@pyrallis.wrap()
def run(train_config: JsrlTrainConfig, seed: int):
    train_config.seed = seed
    train_config.group = train_config.env + "_" + train_config.horizon_fn
    timestr = time.strftime("%d%m%y-%H%M%S")
    train_config.name = f"seed{seed}_{timestr}"
    return evaluate(train_config)

if __name__ == "__main__":
    res = run(0)
