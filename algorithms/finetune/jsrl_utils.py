import torch
import numpy as np
from collections import deque
from pathlib import PosixPath, Path
from goal_horizon_fns import goal_dist_calc
from torch import nn
import guide_heuristics
from variance_learner import StateDepFunction, VarianceLearner
from iql import (
    DeterministicPolicy,
    GaussianPolicy,
    ImplicitQLearning,
    TwinQ,
    ValueFunction
)

horizon_str = ""


def add_jsrl_metrics(eval_log, config):
    eval_log["eval/jsrl/curriculum_stage_idx"] = config.curriculum_stage_idx
    eval_log["eval/jsrl/curriculum_stage"] = config.curriculum_stage
    eval_log["eval/jsrl/best_eval_score"] = config.best_eval_score
    eval_log["eval/jsrl/mean_horizon_reached"] = config.mean_horizon_reached
    eval_log["eval/jsrl/mean_agent_type"] = config.eval_mean_agent_type
    return eval_log


def horizon_update_callback(config, eval_reward):
    #config.rolling_mean_rews.append(eval_reward)
    #rolling_mean = np.mean(config.rolling_mean_rews)
    rolling_mean = eval_reward
    #if config.curriculum_stage == config.all_curriculum_stages[-1]:
    if config.agent_type_stage == 1.0:
        return config
    if not np.isinf(config.best_eval_score):
        prev_best = config.best_eval_score - config.tolerance * config.best_eval_score
    else:
        prev_best = config.best_eval_score

    if (
        rolling_mean >= config.best_eval_score
    ):
        #config.curriculum_stage_idx += 1
        config.agent_type_stage = min(1.0, config.agent_type_stage+config.learner_frac)
        #config.curriculum_stage = config.all_curriculum_stages[
        #    config.curriculum_stage_idx
        #]
        #config.agent_type_stage = config.all_agent_types[config.curriculum_stage_idx]
        config.best_eval_score = rolling_mean
    elif rolling_mean < prev_best:
        config.agent_type_stage = max(config.learner_frac, config.agent_type_stage-config.learner_frac)
        config.best_eval_score = rolling_mean
    #else:
        #config.best_eval_score = rolling_mean
    print(f"curr best: {config.best_eval_score}, rolling mean: {rolling_mean}, agent type: {config.agent_type_stage}")
    return config


def load_guide(trainer, pretrained):
    if not isinstance(pretrained, PosixPath):
        return pretrained
    try:
        trainer.load_state_dict(torch.load(pretrained))
    except RuntimeError:
        trainer.load_state_dict(
            torch.load(pretrained, map_location=torch.device("cpu"))
        )

    guide = trainer.actor
    guide.eval()
    return guide


def prepare_finetuning(init_horizon, mean_return, config):
    #curriculum_stages = HORIZON_FNS[config.horizon_fn]["generate_curriculum_fn"](
    #    init_horizon, config.n_curriculum_stages
    #)
    if config.no_agent_types:
        config.all_agent_types = np.linspace(1, 1, config.n_curriculum_stages)
    else:
        config.all_agent_types = np.linspace(0, 1, config.n_curriculum_stages)
    #config.all_curriculum_stages = curriculum_stages
    config.curriculum_stage_idx = 0
    #config.curriculum_stage = curriculum_stages[config.curriculum_stage_idx]
    #config.agent_type_stage = config.all_agent_types[config.curriculum_stage_idx]
    config.agent_type_stage = config.learner_frac
    if config.n_curriculum_stages == 1:
        config.agent_type_stage = 1
    config.best_eval_score = mean_return
    #config.rolling_mean_rews = deque(maxlen=config.rolling_mean_n)
    return config

def get_var_predictor(env, config, max_steps, guide):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    n_updates = 10000
    if config.horizon_fn == "variance":
        var_actor = guide
        try:
            vf = StateDepFunction(state_dim)
            mf = StateDepFunction(state_dim)
            fn = f"jsrl-CORL/algorithms/finetune/var_functions/{env.unwrapped.spec.name}_guide_{n_updates}_{str(config.variance_learn_frac).replace('.','-')}"
            #fn = "skfhskd"
            vf.load_state_dict(torch.load(fn+"_vf.pt"))
            mf.load_state_dict(torch.load(fn+"_mf.pt"))
            v_learner = VarianceLearner(state_dim, action_dim, config, var_actor)
            v_learner.vf = vf
            v_learner.mf = mf
            v_learner.test_model(env, max_steps, guide)
        except FileNotFoundError:
            vf = VarianceLearner(state_dim, action_dim, config, var_actor).run_training(env, max_steps, guide, n_updates=n_updates, evaluate=True)
        config.vf = vf.eval()
    return config

def make_actor(config, state_dim, action_dim, max_action, device=None, max_steps=None):
    if device is None:
        device = config.device
    q_network = TwinQ(state_dim, action_dim).to(device)
    v_network = ValueFunction(state_dim).to(device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": max_steps,
    }
    return ImplicitQLearning(**kwargs)

def get_guide_agent(config, trainer, state_dim, action_dim, max_action):
    if config.guide_heuristic_fn is not None:
        guide = getattr(guide_heuristics, config.guide_heuristic_fn)
        guide_trainer = None
    elif trainer is None:
        guide_trainer = make_actor(config, state_dim, action_dim, max_action)
        guide = load_guide(guide_trainer, Path(config.pretrained_policy_path))
        guide.eval()
    else:
        guide = trainer.actor
        guide_trainer = trainer
        guide.eval()
    return guide, guide_trainer

def get_learning_agent(config, guide_trainer, init_horizon, mean_return, state_dim, action_dim, max_action):
    trainer = make_actor(config, state_dim, action_dim, max_action)
    if config.n_curriculum_stages == 1 and config.guide_heuristic_fn is None:
        state_dict = guide_trainer.state_dict()
        trainer.partial_load_state_dict(state_dict)
    trainer.total_it = config.offline_iterations # iterations done so far
    config = prepare_finetuning(init_horizon, mean_return, config)
    return trainer, config

def variance_horizon(_, s, _e, config):
    use_learner = False
    var = config.vf(torch.Tensor(s))

    if np.isnan(config.curriculum_stage):
        return True, var
    if (
        (var <= config.curriculum_stage
         or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, var

def timestep_horizon(step, _s, _e, config):
    use_learner = False
    if np.isnan(config.curriculum_stage):
        return True, step
    if (
        (step >= config.curriculum_stage
         or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, step


def goal_distance_horizon(_t, s, env, config):
    use_learner = False
    goal_dist = goal_dist_calc(s, env)
    if np.isnan(config.curriculum_stage):
        return True, goal_dist
    if (
        (goal_dist <= config.curriculum_stage
        or config.curriculum_stage_idx == (config.n_curriculum_stages-1))
        and config.ep_agent_type <= config.agent_type_stage
    ) or (
        goal_dist > config.all_curriculum_stages[-1]
        and config.ep_agent_type <= config.agent_type_stage
    ):
        use_learner = True
    return use_learner, goal_dist


def max_accumulator(v):
    return np.max(v)


def mean_accumulator(v):
    return np.mean(v)


def max_to_min_curriculum(init_horizon, n_curriculum_stages):
    return np.linspace(0, 0, n_curriculum_stages)


def min_to_max_curriculum(init_horizon, n_curriculum_stages):
    return np.linspace(0, init_horizon, n_curriculum_stages)


HORIZON_FNS = {
    "time_step": {
        "horizon_fn": timestep_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": max_to_min_curriculum,
    },
    "goal_dist": {
        "horizon_fn": goal_distance_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    },
    "variance": {
        "horizon_fn": variance_horizon,
        "accumulator_fn": mean_accumulator,
        "generate_curriculum_fn": min_to_max_curriculum,
    }
}


def accumulate(vals):
    return HORIZON_FNS[horizon_str]["accumulator_fn"](vals)


def learner_or_guide_action(state, step, env, learner, guide, config, device, eval=False):
    if guide is None:
        horizon = 0
        use_learner = True
    else:
        if np.random.random() <= config.agent_type_stage:
            use_learner = True
        else:
            use_learner = False
        horizon = 0

    if use_learner:
        # other than the actual learner, this may also be the training guide policy,
        # or the guide being evaluated before online training starts
        if not isinstance(learner, GaussianPolicy):
            action = learner(env, state)
        else:
            if eval:
                action = learner.act(state, device)
            else:
                action = learner(
                    torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
                )
    else:
        if not isinstance(guide, GaussianPolicy):
            action = guide(env, state)
        else:
            action = guide.act(state, device)
        if not eval:
             action = torch.tensor(action)
    return action, use_learner, horizon
