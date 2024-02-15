import numpy as np


def antmaze(_, env):
    goal_state = np.array(env.target_goal)
    current_state = np.array(env.get_xy())
    goal_dist = np.linalg.norm(goal_state-current_state)
    return goal_dist

def lunar_lander(state, _):
    # Compares the x,y pos and whether the lander's
    # legs are touching the ground
    
    goal_state = np.array([0,0,0,0,0,0,1,1])[:2][-2:]
    current_state = state[:2][-2:]
    goal_dist = np.linalg.norm(goal_state-current_state)
    return goal_dist


GOAL_MAP = {"antmaze-umaze-v2": antmaze,
            "antmaze-umaze-diverse-v2": antmaze,
            "antmaze-medium-play-v2": antmaze,
            "antmaze-medium-diverse-v2": antmaze,
            "antmaze-large-play-v2": antmaze,
            "antmaze-large-diverse-v2": antmaze,
            "LunarLander-v2": lunar_lander}

def goal_dist_calc(state, env):
    goal_dist_fn = GOAL_MAP[env.spec.id]
    return goal_dist_fn(state, env)

