from jsrl_utils import make_actor
from jsrl_w_iql import JsrlTrainConfig
from pathlib import Path
import gymnasium
from gymnasium.wrappers import RecordVideo
import torch
from iql import wrap_env
import numpy as np

def evaluate(config: JsrlTrainConfig):
    trigger = lambda t: t%1==0
    state_mean, state_std = 0, 1
    env = RecordVideo(wrap_env(gymnasium.make(config.env, **config.env_config), state_mean, state_std), video_folder="./videos", episode_trigger=trigger, fps=10)
    #env = wrap_env(gymnasium.make(config.env, **config.env_config), state_mean, state_std)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    trainer = make_actor(config, state_dim, action_dim, max_action, device=config.device, max_steps=env.spec.max_episode_steps)
    policy_file = Path(config.load_model)
    trainer.load_state_dict(torch.load(policy_file, map_location=torch.device("cpu")))
    actor = trainer.actor
    actor.eval()
    total_rew = []
    for ep in range(config.n_episodes):
        episode_rew = 0
        if ep == 0:
            try:
                # gym
                env.seed(config.seed+ep)
                state = env.reset()
            except AttributeError:
                # gymnasium
                state, _ = env.reset(seed=config.seed+ep)
            done = False
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
            done = False
        step = 0
        while not done:
            action = actor.act(state, config.device)
            try:
                state, reward, done, env_infos = env.step(action)
            except ValueError:
                state, reward, term, trunc, env_infos = env.step(action)
                done = term or trunc
            episode_rew += reward
            #print(reward)
            env.render()
            step+=1
        print(f"Episode {ep} Reward: {episode_rew}, Steps: {step}")
        total_rew.append(episode_rew)
    env.close()
    print("Mean ep reward: ", np.mean(total_rew))
        