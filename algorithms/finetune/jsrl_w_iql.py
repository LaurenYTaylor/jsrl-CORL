# source: https://github.com/gwthomas/IQL-PyTorch
# IQL: https://arxiv.org/pdf/2110.06169.pdf
# JSRL: https://arxiv.org/abs/2204.02372
#####################################################################################################
# Docstrings were generated using ChatGPT (GPT3.5).                                                 #
# OpenAI. (2024). Docstrings for CombinationLock environment. Retrieved from ChatGPT on 22 May 2024.#
#####################################################################################################
import os
from dataclasses import asdict, dataclass, field
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
import gymnasium_robotics
gymnasium.register_envs(gymnasium_robotics)

from iql import (
    ENVS_WITH_GOAL,
    GaussianPolicy,
    DeterministicPolicy,
    ReplayBuffer,
    TrainConfig,
    Tuple,
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
    n_curriculum_stages: int = 10  # Number of curriculum stages
    tolerance: float = 0.05  # Tolerance level for eval return for proceeding to next curriculum stage
    rolling_mean_n: int = 5  # Number of episodes for rolling mean calculation of eval return
    horizon_fn: str = "time_step"  # Function to determine the horizon type
    new_online_buffer: bool = True  # Whether to create a fresh online replay buffer
    online_buffer_size: int = 10000  # Size of the online buffer
    max_init_horizon: bool = False  # Whether to use the maximum or mean initial horizon (e.g. time step) as curriculum stage 1
    guide_heuristic_fn: str = None  # Name of the guide heuristic function in guide_heuristics.py, if any
    no_agent_types: bool = False  # Whether to restrict agent sampling below a percentage depending on curriculum stage
    variance_learn_frac: float = 0.9  # if horizon=="variance", how often the variance learner should take a random action
    env_config: dict = field(default_factory=lambda: {})  # Environment configuration parameters
    downloaded_dataset: str = None  # Path to downloaded dataset, if any (pre-downloading the dataset is faster)
    pretrained_policy_path: str = None  # Path to pretrained policy file, if any

@torch.no_grad()
def eval_actor(
    env: gym.Env,
    learner: nn.Module,
    guide: nn.Module,
    config: JsrlTrainConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the performance of the combined learner and guide policy in the environment,
    and collect metrics about the performance. This is also called initially just to evaluate
    the performance of the guide policy.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to evaluate the policy in.
    learner : nn.Module
        The learner policy model to be evaluated.
    guide : nn.Module
        The guide policy model used to assist the learner, if applicable.
    config : JsrlTrainConfig
        The configuration parameters for the training and evaluation process.

    Returns
    -------
    episode_rewards : np.ndarray
        An array of total rewards obtained in each episode.
    mean_successes : float
        The mean success rate over all episodes.
    mean_horizon : float
        The mean horizon (e.g. time step, goal distance, variance) reached over all episodes.
    mean_agent_types : float
        The mean type of agent used (learner or guide) over all episodes.
    """
    if isinstance(learner, GaussianPolicy) or isinstance(learner, DeterministicPolicy):
        learner.eval()
    episode_rewards = []
    successes = []
    horizons_reached = []
    agent_types = []
    for i in range(config.n_episodes):
        if i == 0:
            if "gymnasium" not in str(type(env)):
                # gym
                env.seed(config.seed)
                state = env.reset()
            else:
                # gymnasium
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
            
            if config.discrete:
                if use_learner and guide is not None:
                    action = np.argmax(action) 
            episode_horizons.append(horizon)
            if use_learner:
                ep_agent_types.append(1)
            else:
                ep_agent_types.append(0)
                
            if "gymnasium" not in str(type(env)):
                state, reward, done, env_infos = env.step(action)
            else:
                state, reward, term, trunc, env_infos = env.step(action)
                done = term or trunc
            episode_reward += reward
            ts += 1
            
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)

        # Valid only for environments with goal
        print(f"{i}: {episode_reward}")
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

        if guide is None and config.max_init_horizon:
            horizons_reached.append(np.max(episode_horizons))
        else:
            horizons_reached.append(jsrl.accumulate(episode_horizons))

        agent_types.append(np.mean(ep_agent_types))

    if guide is None and config.max_init_horizon:
        # guide is None when guide is first evaluated to determine curruculum stage 1
        # confusingly, during this phase, learner=guide_policy and guide=None
        horizon = np.max(horizons_reached)
    else:
        horizon = np.mean(horizons_reached)
    
    if isinstance(learner, GaussianPolicy) or isinstance(learner, DeterministicPolicy):
        learner.train()
    
    return (
        np.asarray(episode_rewards),
        np.mean(successes),
        horizon,
        np.mean(agent_types),
    )


def jsrl_online_actor(config, env, trainer, max_steps):
    """
    Initialize and configure the online actor for JSRL.

    This function sets up the environment information, selects the guide agent, and configures the learning agent
    based on the initial evaluation of the guide agent. It supports different horizon functions and curriculum stages.

    Parameters
    ----------
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.
    env : gym.Env
        The Gym environment in which the agent will be trained.
    trainer : Any
        The initial trainer object for the learner agent.
    max_steps : int
        The maximum number of steps for the training process.

    Returns
    -------
    trainer : Any
        The configured trainer object for the learner agent.
    guide : nn.Module
        The guide policy model used to assist the learner.
    config : JsrlTrainConfig
        The updated configuration parameters after initialization.
    """
    if config.discrete:
        max_action = 1
        action_dim = env.action_space.n
    else:
        max_action = float(env.action_space.high[0])
        action_dim = env.action_space.shape[0]
    env_info = {"state_dim": env.observation_space.shape[0],
                "action_dim": action_dim,
                "max_action": max_action
                }
    
    config.curriculum_stage = np.nan
    if config.n_curriculum_stages == 1:
        # This is essentially IQL
        init_horizon = 0
        
    guide, guide_trainer = jsrl.get_guide_agent(config, trainer, **env_info)
    if config.horizon_fn == "variance":
        config = jsrl.get_var_predictor(env, config, max_steps, guide)
    _, _, init_horizon, _ = eval_actor(env, guide, None, config)
    trainer, config = jsrl.get_learning_agent(config, guide_trainer, init_horizon, **env_info)
    return trainer, guide, config

def get_online_buffer(config, replay_buffer, state_dim, action_dim):
    """
    Initialize or reuse an online replay buffer for experience replay during training.

    Parameters
    ----------
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process, including buffer settings.
    replay_buffer : ReplayBuffer or None
        The existing replay buffer, if any. This buffer may be reused or replaced based on the configuration.
    state_dim : int
        The dimensionality of the state space.
    action_dim : int
        The dimensionality of the action space.

    Returns
    -------
    online_replay_buffer : ReplayBuffer
        The initialized or reused online replay buffer.
    """
    if config.new_online_buffer:
        if replay_buffer is not None:
            del replay_buffer
        online_replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            config.online_buffer_size,
            config.device,
        )
    else:
        online_replay_buffer = replay_buffer
    return online_replay_buffer

def process_minari_data(downloaded_data):
    rearranged_data = {"actions": [], "infos": [], "observations": [],
                       "rewards": [], "terminals": [], "timeouts": []}
    for k,v in downloaded_data.items():
        rearranged_data["observations"].extend(v['observations'])
        rearranged_data["actions"].extend(v['actions'])
        rearranged_data["infos"].extend(v['infos'])
        rearranged_data["rewards"].extend(v['rewards'])
        rearranged_data["timeouts"].extend(v['truncations'])
        rearranged_data["terminals"].extend(v['terminations'])
    for k,v in rearranged_data.items():
        rearranged_data[k] = np.array(v)
    return rearranged_data

def train(config: JsrlTrainConfig):
    """
    Train an learning agent using JSRL method (gradual online transfer from pre-trained guide agent to learner).
    Also handles offline pre-training for D4RL environments.
    Mostly a replica of the IQL code in the same file, with JSRL function calls added.

    Parameters
    ----------
    config : JsrlTrainConfig
        The configuration parameters for the JSRL training process.

    Returns
    -------
    None
        The agent's model is saved periodically.
    """
    
    # Added functionality for handling both Gym/Gymnasium envs
    try:
        env = gymnasium.make(config.env, **config.env_config)
        eval_env = gymnasium.make(config.env, **config.env_config)
        max_steps = env.spec.max_episode_steps
    except:
        env = gym.make(config.env, **config.env_config)
        eval_env = gym.make(config.env, **config.env_config)
        max_steps = env._max_episode_steps

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    state_dim = env.observation_space.shape[0]
    
    if (type(env.action_space) == gym.spaces.Discrete or
    type(env.action_space) == gymnasium.spaces.Discrete):
        action_dim = env.action_space.n
        config.discrete = True
    else:
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
        if 'episode_1' in downloaded_data.keys():
            downloaded_data = process_minari_data(downloaded_data)
        dataset = d4rl.qlearning_dataset(env, dataset=downloaded_data)
    elif config.guide_heuristic_fn is None:
        dataset = d4rl.qlearning_dataset(env)
    else:
        dataset = None #  a heuristic is being used as the guide
        config.normalize_reward = False
        config.normalize = False

    replay_buffer = None
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

    if config.discrete:
        max_action = 1
    else:
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
    except AttributeError:
        pass


    print("----------------------------------------------------")
    print(f"Training IQL+JSRL, Env: {config.env}, Seed: {seed}")
    print("----------------------------------------------------")

    # Create the IQL-based offline learner, or load a pre-trained guide
    # (or do nothing if guide is a heuristic)
    if config.pretrained_policy_path is None and config.guide_heuristic_fn is None:
        trainer = jsrl.make_actor(config, state_dim, action_dim, max_action, max_steps=config.offline_iterations)
        if config.load_model != "":
            policy_file = Path(config.load_model)
            trainer.load_state_dict(torch.load(policy_file))
            actor = trainer.actor
        else:
            actor = trainer.actor
    else:
        trainer = None
        actor = None
    
    # Initialise WANDB    
    wandb_init(asdict(config))
        
    if "gymnasium" not in str(type(env)):
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
        config.offline_iterations = 0

    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
            if config.guide_heuristic_fn is not None:
                actor = getattr(guide_heuristics, config.guide_heuristic_fn)
            # Make the online learning agent, return the pre-trained or loaded guide policy
            trainer, guide, config = jsrl_online_actor(config, env, trainer, max_steps)
            actor = trainer.actor
            
            # Create the new online buffer
            online_replay_buffer = get_online_buffer(config, replay_buffer, state_dim, action_dim)

        online_log = {}
        if t >= config.offline_iterations:
            # ep_agent_type == 1 -> 100% use of learner during ep
            # ep_agent_type == 0 -> 100% use of guide during ep
            if episode_step == 0:
                   # Initialises env
                episode_agent_types = []
                config.ep_agent_type = 0
            else:
                config.ep_agent_type = np.mean(episode_agent_types)


            action, use_learner, _ = jsrl.learner_or_guide_action(
                state,
                episode_step,
                env,
                actor,
                guide,
                config,
                config.device,
            )

            if use_learner:
                episode_agent_types.append(1)
                buffer_action = action.cpu().data.numpy().flatten()
                if config.discrete:
                    action = torch.argmax(action)
                elif not config.iql_deterministic:
                    action = action.sample()
                else:
                    noise = (torch.randn_like(action) * config.expl_noise).clamp(
                        -config.noise_clip, config.noise_clip
                    )
                    action += noise
            else:
                episode_agent_types.append(0)
                if config.discrete:
                    buffer_action = np.zeros(action_dim,dtype=float)
                    buffer_action[action] = 1.0
            if not config.discrete:
                action = torch.clamp(max_action * action, -max_action, max_action)
                buffer_action = action
            action = action.cpu().data.numpy().flatten()
            
            if config.discrete:
                action = action[0]
            if "gymnasium" not in str(type(env)):
                next_state, reward, done, env_infos = env.step(action)
            else:
                next_state, reward, term, trunc, env_infos = env.step(action)
                done = term or trunc
            episode_step += 1
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward

            real_done = False  # Episode can timeout which is different from done for gym
            if done and episode_step < max_steps:
                real_done = True
            
            #print(f"{episode_step} - term: {term}, trunc: {trunc}, done: {done}, real done: {real_done}")
            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)

            #print(buffer_action)
            online_replay_buffer.add_transition(
                state, buffer_action, reward, next_state, real_done
            )
            state = next_state
            if done:
                if "gymnasium" not in str(type(env)):
                    state = env.reset()
                else:
                    state, _ = env.reset()
                done = False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                print(f"{t}: {episode_return}")
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

        # Do an update once their are batch_size samples in buffer
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
                if t < config.offline_iterations:
                    guide = None
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
                if is_env_with_goal:
                    eval_successes.append(success_rate)
                    eval_log["eval/regret"] = np.mean(1 - np.array(eval_successes))
                    eval_log["eval/success_rate"] = success_rate
                if t >= config.offline_iterations:
                    config = jsrl.horizon_update_callback(config, normalized)
                    eval_log = jsrl.add_jsrl_metrics(eval_log, config)
                if config.normalize_reward:
                    normalized_eval_score = normalized * 100.0
                    eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
                else:
                    eval_log["eval/score"] = normalized
                #print("---------------------------------------")
                eval_str = f"Evaluation over {config.n_episodes} episodes: "\
                    f"{eval_score:.3f}"
                if config.normalize_reward:
                    eval_str += " , D4RL score: {normalized_eval_score:.3f}"
                #print(eval_str)
                #print("---------------------------------------")
                if config.checkpoints_path is not None:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                    )
                wandb.log(eval_log, step=trainer.total_it)
    wandb.finish(exit_code=0)


if __name__ == "__main__":
    train(pyrallis.parse(config_class=JsrlTrainConfig))
