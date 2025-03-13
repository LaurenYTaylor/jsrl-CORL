import pyrallis
import os
from jsrl_w_iql import JsrlTrainConfig, train
import time
import wandb

@pyrallis.wrap()
def run(train_config: JsrlTrainConfig, seed: int):
    train_config.seed = seed
    train_config.group = train_config.env + "_" + train_config.horizon_fn
    timestr = time.strftime("%d%m%y-%H%M%S")
    train_config.name = f"seed{seed}_{timestr}"
    data_path = "jsrl-CORL/downloaded_data/" + train_config.env + ".hdf5"
    if os.path.exists(data_path):
        train_config.downloaded_dataset = data_path
    return train(train_config)

if __name__ == "__main__":
    extra_config = {}
    seeds = range(1)
    for seed in seeds:
        res = run(seed)
   