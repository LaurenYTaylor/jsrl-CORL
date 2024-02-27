# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import os
import sys
from dataclasses import asdict, dataclass, field
from collections import defaultdict

from pathlib import Path

import d4rl
import gym
import gymnasium
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import wandb
import h5py
from gymnasium.wrappers import StepAPICompatibility


from iql import (
    DeterministicPolicy,
    ENVS_WITH_GOAL,
    GaussianPolicy,
    ImplicitQLearning,
    ReplayBuffer,
    TrainConfig,
    Tuple,
    TwinQ,
    ValueFunction,
    compute_mean_std,
    is_goal_reached,
    modify_reward,
    modify_reward_online,
    nn,
    normalize_states,
    set_env_seed,
    set_seed,
    wandb_init,
    wrap_env,
)
import jsrl_utils as jsrl
import guide_heuristics as guide_heuristics


@dataclass(kw_only=True)
class JsrlTrainConfig(TrainConfig):
    n_curriculum_stages: int = 10
    tolerance: float = 0.05
    rolling_mean_n: int = 5
    pretrained_policy_path: str = None
    horizon_fn: str = "time_step"
    downloaded_dataset: str = None
    new_online_buffer: bool = True
    online_buffer_size: int = 10000
    max_init_horizon: bool = True
    env_config: dict = field(default_factory= lambda: {})
    guide_heuristic_fn: str = None


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    learner: nn.Module,
    guide: nn.Module,
    config: JsrlTrainConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(learner, GaussianPolicy):
        learner.eval()
    episode_rewards = []
    successes = []
    horizons_reached = []
    agent_types = []
    for i in range(config.n_episodes):
        if i == 0:
            try:
                env.seed(config.seed)
                state = env.reset()
            except AttributeError:
                state, _ = env.reset(seed=config.seed)
            done = False
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
            done = False
        
        ts = 0
        episode_reward = 0.0
        episode_horizons = []
        ep_agent_types = []
        goal_achieved = False
        while not done:
            if ts == 0:
                config.ep_agent_type = 0
            else:
                config.ep_agent_type = np.mean(ep_agent_types)
            action, use_learner, horizon = jsrl.learner_or_guide_action(
                state, ts, env, learner, guide, config, config.device, eval=True
            )
            episode_horizons.append(horizon)
            if use_learner:
                ep_agent_types.append(1)
            else:
                ep_agent_types.append(0)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            ts += 1
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

        if guide is None and config.max_init_horizon:
            horizons_reached.append(np.max(episode_horizons))
        else:
            horizons_reached.append(jsrl.accumulate(episode_horizons))

        agent_types.append(np.mean(ep_agent_types))

    if guide is None and config.max_init_horizon:
        horizon = np.max(horizons_reached)
    else:
        horizon = np.mean(horizons_reached)
    
    if isinstance(learner, GaussianPolicy):
        learner.train()
    
    return (
        np.asarray(episode_rewards),
        np.mean(successes),
        horizon,
        np.mean(agent_types),
    )

def make_actor(config: JsrlTrainConfig, state_dim, action_dim, max_action, device=None, max_steps=None):
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
    return kwargs

def train(config: JsrlTrainConfig):
    try:
        env = StepAPICompatibility(gymnasium.make(config.env, **config.env_config), output_truncation_bool=False)
        eval_env = StepAPICompatibility(gymnasium.make(config.env, **config.env_config), output_truncation_bool=False)
        max_steps = env.spec.max_episode_steps
    except:
        env = gym.make(config.env, **config.env_config)
        eval_env = gym.make(config.env, **config.env_config)
        max_steps = env._max_episode_steps

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.downloaded_dataset:
        downloaded_data = {}

        def get_keys(f, dataset):
            for k in f.keys():
                try:
                    dataset[k] = f[k][()]
                except TypeError:
                    dataset[k] = get_keys(f[k], {})
            return dataset

        with h5py.File(config.downloaded_dataset, "r") as f:
            downloaded_data = get_keys(f, {})
        dataset = d4rl.qlearning_dataset(env, dataset=downloaded_data)
    elif config.guide_heuristic_fn is None:
        dataset = d4rl.qlearning_dataset(env)
    else:
        dataset = None
        config.normalize_reward = False
        config.normalize = False

    if dataset is not None:
        reward_mod_dict = {}
        if config.normalize_reward:
            reward_mod_dict = modify_reward(dataset, config.env)

        if config.normalize:
            state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        else:
            state_mean, state_std = 0, 1

        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
        env = wrap_env(env, state_mean=state_mean, state_std=state_std)
        eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
        replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            config.device,
        )
        replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, None)
    try:
        set_env_seed(env, config.eval_seed)
        set_env_seed(eval_env, config.eval_seed)
        seed_set = True
    except AttributeError:
        seed_set = False


    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    if config.pretrained_policy_path is None and config.guide_heuristic_fn is None:
        training_kwargs = make_actor(config, state_dim, action_dim, max_action, max_steps=config.offline_iterations)
        trainer = ImplicitQLearning(**training_kwargs)
        if config.load_model != "":
            policy_file = Path(config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
            actor = trainer.actor
        else:
            actor = trainer.actor
    
    wandb_init(asdict(config))

    evaluations = []

    if seed_set:
        state, done = env.reset(), False
    else:
        state, _ = env.reset(seed=seed)
        done = False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    jsrl.horizon_str = config.horizon_fn

    if config.pretrained_policy_path is not None:
        if config.guide_heuristic_fn:
            sys.exit("Guide can be a pretrained policy OR a heuristic,\
                        but you have provided both. Please choose one.")
        training_kwargs = make_actor(config, state_dim, action_dim, max_action)
        guide_trainer = ImplicitQLearning(**training_kwargs)
        guide = jsrl.load_guide(guide_trainer, Path(config.pretrained_policy_path))
        guide.eval()
        config.curriculum_stage = np.nan
        _, _, init_horizon, _ = eval_actor(env, guide, None, config)
        if config.n_curriculum_stages == 1:
            init_horizon = 0
        config = jsrl.prepare_finetuning(init_horizon, config)
        config.offline_iterations = 0
        kwargs = make_actor(config, state_dim, action_dim, max_action)
        trainer = ImplicitQLearning(**kwargs)
        if config.n_curriculum_stages == 1:
            state_dict = guide_trainer.state_dict()
            trainer.partial_load_state_dict(state_dict)
        kwargs["max_steps"] = config.offline_iterations
        actor = trainer.actor
        trainer.total_it = config.offline_iterations
    else:
        guide = None

    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
            if config.pretrained_policy_path is None:
                config.curriculum_stage = np.nan
                if config.guide_heuristic_fn is not None:
                    actor = getattr(guide_heuristics, config.guide_heuristic_fn)
                    
                from variance_learner import VarianceLearner
                vl = VarianceLearner(state_dim, action_dim, config, None)
                vl.run_training(eval_env, max_steps)
                var_est = vl.varf
                return
                _, _, init_horizon, _ = eval_actor(env, actor, guide, config)
                if config.n_curriculum_stages == 1:
                    init_horizon = 0
                if config.guide_heuristic_fn is not None:
                    guide = getattr(guide_heuristics, config.guide_heuristic_fn)
                else:
                    guide = trainer.actor
                    guide_trainer = trainer
                    #del trainer
                    guide.eval()
                kwargs = make_actor(config, state_dim, action_dim, max_action)
                trainer = ImplicitQLearning(**kwargs)
                if config.n_curriculum_stages == 1 and config.guide_heuristic_fn is None:
                    state_dict = guide_trainer.state_dict()
                    trainer.partial_load_state_dict(state_dict)
                trainer.total_it = config.offline_iterations # iterations done so far
                actor = trainer.actor
                actor.train()
                config = jsrl.prepare_finetuning(init_horizon, config)
            if config.new_online_buffer:
                try:
                    del replay_buffer
                except UnboundLocalError:
                    pass
                online_replay_buffer = ReplayBuffer(
                    state_dim,
                    action_dim,
                    config.online_buffer_size,
                    config.device,
                )

            else:
                online_replay_buffer = replay_buffer

        online_log = {}
        if t >= config.offline_iterations:
            if episode_step == 0:
                episode_agent_types = []
                config.ep_agent_type = 0
            else:
                config.ep_agent_type = np.mean(episode_agent_types)

            episode_step += 1

            action, use_learner, _ = jsrl.learner_or_guide_action(
                state,
                episode_step,
                env,
                actor,
                guide,
                config,
                kwargs["device"],
            )

            if use_learner:
                episode_agent_types.append(1)
                if not config.iql_deterministic:
                    action = action.sample()
                else:
                    noise = (torch.randn_like(action) * config.expl_noise).clamp(
                        -config.noise_clip, config.noise_clip
                    )
                    action += noise
            else:
                episode_agent_types.append(0)

            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)

            online_replay_buffer.add_transition(
                state, action, reward, next_state, real_done
            )
            state = next_state
            if done:
                if seed_set:
                    state, done = env.reset(), False
                else:
                    # just use seed_set to know if it's a gymnasium env,
                    # don't actually need to reseed
                    state, _ = env.reset()
                    done = False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                online_log["train/mean_ep_agent_type"] = np.mean(episode_agent_types)
                if config.normalize_reward:
                    normalized_return = eval_env.get_normalized_score(episode_return)
                    online_log["train/d4rl_normalized_episode_return"] = (
                        normalized_return * 100.0
                    )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        if (
            t >= config.batch_size and config.new_online_buffer
        ) or not config.new_online_buffer:
            if t >= config.offline_iterations:
                batch = online_replay_buffer.sample(config.batch_size)
            elif t < config.offline_iterations:
                batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            log_dict[
                "offline_iter" if t < config.offline_iterations else "online_iter"
            ] = (t if t < config.offline_iterations else t - config.offline_iterations)

            log_dict.update(online_log)
            wandb.log(log_dict, step=trainer.total_it)
            # Evaluate episode
            if (t + 1) % config.eval_freq == 0:
                print(f"Time steps: {t + 1}")
                if guide is None:
                    config.curriculum_stage = np.nan
                else:
                    config.curriculum_stage = config.curriculum_stage
                (
                    eval_scores,
                    success_rate,
                    config.mean_horizon_reached,
                    config.eval_mean_agent_type,
                ) = eval_actor(eval_env, actor, guide, config)

                eval_score = eval_scores.mean()
                eval_log = {}
                if config.normalize_reward:
                    normalized = eval_env.get_normalized_score(eval_score)
                else:
                    normalized = eval_score

                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if t >= config.offline_iterations:
                    if is_env_with_goal:
                        eval_successes.append(success_rate)
                        eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                        eval_log["eval/success_rate"] = success_rate

                    config = jsrl.horizon_update_callback(config, normalized)
                    eval_log = jsrl.add_jsrl_metrics(eval_log, config)
                if config.normalize_reward:
                    normalized_eval_score = normalized * 100.0
                    evaluations.append(normalized_eval_score)
                    eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
                else:
                    eval_log["eval/score"] = normalized
                print("---------------------------------------")
                eval_str = f"Evaluation over {config.n_episodes} episodes: "\
                    f"{eval_score:.3f}"
                if config.normalize_reward:
                    eval_str += " , D4RL score: {normalized_eval_score:.3f}"
                print(eval_str)
                print("---------------------------------------")
                if config.checkpoints_path is not None:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                    )
                wandb.log(eval_log, step=trainer.total_it)


if __name__ == "__main__":
    train(pyrallis.parse(config_class=JsrlTrainConfig))
