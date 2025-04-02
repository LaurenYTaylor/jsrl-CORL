import pyrallis
import os
from jsrl_w_iql import JsrlTrainConfig, train_dict
import time
import wandb
from ray import tune
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
from dataclasses import asdict


@pyrallis.wrap()
def get_config(train_config: JsrlTrainConfig, extra_config: dict, seed: int):
    train_config.seed = seed
    train_config.group = train_config.env + "_" + train_config.horizon_fn
    train_config.project = "debug"
    timestr = time.strftime("%d%m%y-%H%M%S")
    train_config.name = f"seed{seed}_{timestr}"
    data_path = "/workspace/jsrl-CORL/downloaded_data/" + train_config.env + ".hdf5"
    if os.path.exists(data_path):
        train_config.downloaded_dataset = data_path
    return train_config

def run_training(ray_config):
    train_dict(ray_config)

if __name__ == "__main__":
    extra_config = {}
    #"iql_tau": tune.sample_uniform(0.5, 0.99),
    #"beta": tune.sample_uniform(1, 5),
    #"actor_lr": tune.sample_loguniform(1e-4, 1e-2)}
    config_dict = asdict(get_config(extra_config, seed=0))
    #config_dict["eval_freq"] = tune.randint(15000, 25000)
    config_dict["eval_freq"] = tune.randint(15000, 25000)
    config_dict["n_episodes"] = tune.randint(50, 150)
    config_dict["tolerance"] = tune.uniform(0.001, 0.03)
    tuner = tune.Tuner(
        run_training,
        tune_config=tune.TuneConfig(
            num_samples=50,
            scheduler=ASHAScheduler(metric="eval_return", mode="max"),
        ),
        param_space=config_dict,
        run_config=tune.RunConfig(storage_path=Path("./hyperparam_results").resolve(), name="test_experiment")
    )
    results = tuner.fit()
    '''dfs = {result.path: result.metrics_dataframe for result in results}
    for d in dfs.values():
        ax = d.eval_return.plot()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Return")
        plt.savefig("ray_result.png")   
        '''