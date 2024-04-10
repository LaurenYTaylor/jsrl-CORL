import gymnasium as gym
import numpy as np
from copy import deepcopy
import sys

env = gym.make(
    "MiniGrid-DoorKey-16x16-v0",
    render_mode="human"
)


seed = 0
all_ep_rews = []
for ep in range(50):
    term = False
    trunc = False
    if ep==0:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    ep_reward = 0
    while not (term or trunc):
        env.render()
        #print(dict(zip(obs_keys, obs)))
        #goal_dist = np.array([0,0,0,0,0,0,1,1])
        #curr_goal_dist = np.linalg.norm(np.array(obs)[:2][-2:]-goal_dist[:2][-2:])
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        print(obs)
        ep_reward += reward
    print(f"{ep}: {ep_reward}")
    all_ep_rews.append(ep_reward)
print(f"20 Eps Mean Rew: {np.round(np.mean(all_ep_rews), 2)} "\
      f"+/- {np.round(np.std(all_ep_rews), 2)}")