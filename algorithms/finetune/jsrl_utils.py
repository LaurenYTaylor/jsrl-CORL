import torch
import numpy as np
from collections import deque

horizon_str = ""


def add_jsrl_metrics(eval_log, config):
    eval_log["eval/jsrl/curriculum_stage_idx"] = config.curriculum_stage_idx
    eval_log["eval/jsrl/curriculum_stage"] = config.curriculum_stage
    eval_log["eval/jsrl/best_eval_score"] = config.best_eval_score
    eval_log["eval/jsrl/mean_horizon_reached"] = config.mean_horizon_reached
    eval_log["eval/jsrl/mean_agent_type"] = config.mean_agent_type
    return eval_log


def horizon_update_callback(config, eval_reward):
    config.rolling_mean_rews.append(eval_reward)
    rolling_mean = np.mean(config.rolling_mean_rews)
    if config.curriculum_stage == config.all_curriculum_stages[-1]:
        return config
    if not np.isinf(config.best_eval_score):
        prev_best = config.best_eval_score - config.tolerance * config.best_eval_score
    else:
        prev_best = config.best_eval_score

    if (
        len(config.rolling_mean_rews) == config.rolling_mean_n
        and rolling_mean > prev_best
    ):
        config.curriculum_stage_idx += 1
        config.curriculum_stage = config.all_curriculum_stages[
            config.curriculum_stage_idx
        ]
        config.best_eval_score = rolling_mean
    return config


def load_guide(trainer, pretrained):
    trainer.load_state_dict(torch.load(pretrained))
    guide = trainer.actor
    guide.eval()
    return guide


def prepare_finetuning(init_horizon, config):
    curriculum_stages = HORIZON_FNS[config.horizon_fn]["generate_curriculum_fn"](
        init_horizon, config.n_curriculum_stages
    )
    config.all_agent_types = np.linspace(0, 1, config.n_curriculum_stages)
    config.all_curriculum_stages = curriculum_stages
    config.curriculum_stage_idx = 0
    config.curriculum_stage = curriculum_stages[config.curriculum_stage_idx]
    config.agent_type = config.all_agent_types[config.curriculum_stage_idx]
    config.best_eval_score = -np.inf
    config.rolling_mean_rews = deque(maxlen=config.rolling_mean_n)
    config.offline_iterations = 0
    return config


def timestep_horizon(step, _s, _e, curriculum_stage):
    use_learner = False
    if step >= curriculum_stage:
        use_learner = True
    return use_learner, step


def goal_distance_horizon(_t, _s, env, curriculum_stage):
    use_learner = False
    goal_dist = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
    if goal_dist < curriculum_stage:
        use_learner = True
    return use_learner, goal_dist


def max_accumulator(v):
    return np.max(v)


def mean_accumulator(v):
    return np.mean(v)


def max_to_min_curriculum(init_horizon, n_curriculum_stages):
    return np.linspace(init_horizon, 0, n_curriculum_stages)


def min_to_max_curriculum(init_horizon, n_curriculum_stages):
    return np.linspace(0, init_horizon, n_curriculum_stages)


HORIZON_FNS = {
    "time_step": {
        "horizon_fn": timestep_horizon,
        "accumulator_fn": max_accumulator,
        "generate_curriculum_fn": max_to_min_curriculum,
    },
    "goal_dist": {
        "horizon_fn": goal_distance_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
}


def accumulate(vals):
    return HORIZON_FNS[horizon_str]["accumulator_fn"](vals)


def learner_or_guide_action(state, step, env, learner, guide, curriculum_stage, device):
    if guide is None:
        _, horizon = HORIZON_FNS[horizon_str]["horizon_fn"](
            step, state, env, curriculum_stage
        )
        use_learner = True
    else:
        use_learner, horizon = HORIZON_FNS[horizon_str]["horizon_fn"](
            step, state, env, curriculum_stage
        )

    if use_learner:
        action = learner(
            torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        )
    else:
        action = guide(
            torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        )
    return action, use_learner, horizon
