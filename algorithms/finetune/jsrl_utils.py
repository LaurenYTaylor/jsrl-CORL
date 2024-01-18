import torch
import numpy as np
from collections import deque

horizon_str = ""

def add_jsrl_metrics(eval_log, jsrl_config):
    eval_log["eval/jsrl/curriculum_stage_idx"] = jsrl_config["curriculum_stage_idx"]
    eval_log["eval/jsrl/curriculum_stage"] = jsrl_config["curriculum_stage"]
    eval_log["eval/jsrl/best_eval_score"] = jsrl_config["best_eval_score"]
    eval_log["eval/jsrl/mean_horizon_reached"] = jsrl_config["mean_horizon_reached"]
    eval_log["eval/jsrl/mean_agent_type"] = jsrl_config["mean_agent_type"]
    return eval_log

def horizon_update_callback(jsrl_config, eval_reward):
    jsrl_config["rolling_mean_rews"].append(eval_reward)
    rolling_mean = np.mean(jsrl_config["rolling_mean_rews"])
    if not np.isinf(jsrl_config["best_eval_score"]):
        prev_best = (
            jsrl_config["best_eval_score"]
            - jsrl_config["tolerance"]
            * jsrl_config["best_eval_score"]
        )
    else:
        prev_best = jsrl_config["best_eval_score"]
        
    if (
        len(jsrl_config["rolling_mean_rews"])
        == jsrl_config["rolling_mean_n"]
        and rolling_mean > prev_best
    ):
        jsrl_config["curriculum_stage_idx"] += 1
        jsrl_config["curriculum_stage"] = jsrl_config["all_curriculum_stages"][jsrl_config["curriculum_stage_idx"]]
        jsrl_config["best_eval_score"] = rolling_mean
    return jsrl_config

def load_guide(trainer, pretrained):
    trainer.load_state_dict(torch.load(pretrained))
    guide = trainer.actor
    guide.eval()
    return guide
    
def prepare_finetuning(init_horizon, config, jsrl_config):
    curriculum_stages = np.linspace(init_horizon, 0, jsrl_config["n_curriculum_stages"])
    jsrl_config["all_curriculum_stages"] = curriculum_stages 
    jsrl_config["curriculum_stage_idx"] = 0
    jsrl_config["curriculum_stage"] = curriculum_stages[jsrl_config["curriculum_stage_idx"]]
    jsrl_config["best_eval_score"] = -np.inf
    jsrl_config["rolling_mean_rews"] = deque(maxlen=jsrl_config["rolling_mean_n"])
    config.offline_iterations = 0
    config.jsrl_config = jsrl_config
    return config

def timestep_horizon(step, _, curriculum_stage):
    action_policy = "guide"
    if (step >= curriculum_stage):
        action_policy = "learning"
    return action_policy, step

def max_accumulator(v):
    return np.max(v)

def mean_accumulator(v):
    return np.mean(v)

HORIZON_FNS = {"time_step": {"horizon_fn": timestep_horizon,
                             "accumulator_fn": max_accumulator}}

def accumulate(vals):
    return HORIZON_FNS[horizon_str]["accumulator_fn"](vals)

def learner_or_guide_action(state, step, learner, guide, curriculum_stage, device):
    
    use_learner, horizon = HORIZON_FNS[horizon_str]["horizon_fn"](step, state, curriculum_stage)
    
    if use_learner:
        action = learner(
            torch.tensor(
                state.reshape(1, -1), device=device, dtype=torch.float32
            )
        )
    else:
        action = guide(
            torch.tensor(
                state.reshape(1, -1), device=device, dtype=torch.float32
            )
        )
    return action, use_learner, horizon
