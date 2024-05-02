import numpy as np

def combination_lock(env, _):
    next_number = env.unwrapped.combination[env.combo_step]
    action = int(next_number)
    return action

def lunar_lander(env, state):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            state[0] is the horizontal coordinate
            state[1] is the vertical coordinate
            state[2] is the horizontal speed
            state[3] is the vertical speed
            state[4] is the angle
            state[5] is the angular speed
            state[6] 1 if first leg has contact, else 0
            state[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = state[0] * 0.5 + state[2] * 1.0  # angle should point towards center
    if angle_targ > 0.8:
        angle_targ = 0.8  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.8:
        angle_targ = -0.8
    hover_targ = 0.55 * np.abs(
        state[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
    hover_todo = (hover_targ - state[1]) * 0.25 - (state[3]) * 0.25

    '''
    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(state[3]) * 0.5
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

def lunar_lander_perfect(env, state):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            state[0] is the horizontal coordinate
            state[1] is the vertical coordinate
            state[2] is the horizontal speed
            state[3] is the vertical speed
            state[4] is the angle
            state[5] is the angular speed
            state[6] 1 if first leg has contact, else 0
            state[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = state[0] * 0.5 + state[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        state[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
    hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5

    if state[6] or state[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(state[3]) * 0.5
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