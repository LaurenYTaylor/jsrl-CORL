from algorithms.finetune.jsrl_wrapper import JsrlTrainConfig
import pyrallis
import ray
from train_online import main, make_save_dir
from memory_profiler import profile


@ray.remote
def run_training(seed, n_data, save_dir, config, dataset_name):
    config["seed"] = seed
    config["init_dataset_size"] = n_data
    config["save_dir"] = save_dir
    config["downloaded_dataset"] = f"datasets/{dataset_name}.pkl"
    print(config)
    return main(config)


@pyrallis.wrap
def run(train_config, extra_config):
    object_references = [
        (
            run_training.options(
                num_cpu=1, num_gpu=extra_config["gpu_frac"]
            ).run_training.remote(seed, train_config)
        )
        for seed in extra_config["seeds"]
    ]

    all_data = []
    while len(object_references) > 0:
        finished, object_references = ray.wait(object_references, timeout=7.0)
        data = ray.get(finished)
        all_data.extend(data)


if __name__ == "__main__":
    extra_config = {}

    extra_config["seeds"] = range(4)
    extra_config["goals"] = ["time_step", "goal_dist"]
    num_cpus = len(extra_config["seeds"]) * len(extra_config["goals"])
    extra_config["gpu_frac"] = 2 / num_cpus

    ray.init(num_cpus=num_cpus, num_gpus=2)

    run(extra_config)
