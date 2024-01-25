import pyrallis
import ray
import os
from jsrl_wrapper import JsrlTrainConfig, train


@ray.remote
def run_training(seed, goal, train_config):
    train_config.seed = seed
    train_config.horizon_fn = goal
    train_config.group = train_config.env + "_" + train_config.horizon_fn
    train_config.name = f"seed{seed}"
    data_path = "jsrl-CORL/downloaded_data/" + train_config.env + ".hdf5"
    if os.path.exists(data_path):
        train_config.downloaded_dataset = data_path
    return train(train_config)


@pyrallis.wrap()
def run(train_config: JsrlTrainConfig, extra_config: dict):
    rt_w_options = run_training.options(num_gpus=extra_config["gpu_frac"])
    object_references = [
        rt_w_options.remote(seed, goal, train_config)
        for seed in extra_config["seeds"]
        for goal in extra_config["goals"]
    ]

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(object_references, timeout=7.0)
        data = ray.get(finished)
        all_data.extend(data)


if __name__ == "__main__":
    extra_config = {}

    extra_config["seeds"] = range(4)
    extra_config["goals"] = ["goal_dist"]
    extra_config["gpu_frac"] = (
        1 / (len(extra_config["seeds"]) * len(extra_config["goals"]))
    )
    print(extra_config["gpu_frac"])

    run(extra_config)
