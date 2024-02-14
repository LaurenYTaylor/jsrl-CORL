import pyrallis
import ray
import os
from jsrl_wrapper import JsrlTrainConfig, train
import time


@ray.remote
def run_training(seed, train_config):
    train_config.seed = seed
    train_config.group = train_config.env + "_" + train_config.horizon_fn
    timestr = time.strftime("%d%m%y-%H%M%S")
    train_config.name = f"seed{seed}_{timestr}"
    data_path = "jsrl-CORL/downloaded_data/" + train_config.env + ".hdf5"
    if os.path.exists(data_path):
        train_config.downloaded_dataset = data_path
    return train(train_config)


@pyrallis.wrap()
def run(train_config: JsrlTrainConfig, extra_config: dict):
    rt_w_options = run_training.options(num_gpus=extra_config["gpu_frac"])
    object_references = [
        rt_w_options.remote(seed, train_config) for seed in extra_config["seeds"]
    ]

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(object_references, timeout=7.0)
        data = ray.get(finished)
        all_data.extend(data)


if __name__ == "__main__":
    extra_config = {}

    extra_config["seeds"] = range(4)
    extra_config["gpu_frac"] = 1 / (len(extra_config["seeds"]))

    run(extra_config)
