import gymnasium as gym
import numpy as np
from copy import deepcopy
import sys

env = gym.make(
    "LunarLander-v2",
    continuous = True,
    turbulence_power=2,
    enable_wind=True,
    wind_power=20
)
def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

def imperfect_heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.8:
        angle_targ = 0.8  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.8:
        angle_targ = -0.8
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.2 - (s[3]) * 0.2

    '''
    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact
    '''

    if env.continuous:
        a = np.array([hover_todo * 15 - 1, -angle_todo * 15])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a

obs_keys = ["x_coord", "y_coord", "x_lin_vel", "y_lin_vel", "angle",
            "ang_vel", "right_leg_contact", "left_leg_contact"]
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
        #print(dict(zip(obs_keys, obs)))
        goal_dist = np.array([0,0,0,0,0,0,1,1])
        curr_goal_dist = np.linalg.norm(np.array(obs)[:2][-2:]-goal_dist[:2][-2:])
        action = imperfect_heuristic(env, obs)
        obs, reward, term, trunc, _ = env.step(action)
        
        #env.render()
        ep_reward += reward
        seed += 4
    print(f"{ep}: {ep_reward}")
    all_ep_rews.append(ep_reward)
print(f"20 Eps Mean Rew: {np.round(np.mean(all_ep_rews), 2)} "\
      f"+/- {np.round(np.std(all_ep_rews), 2)}")