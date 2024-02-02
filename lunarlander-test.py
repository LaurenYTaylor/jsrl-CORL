import gymnasium as gym

env = gym.make(
    "LunarLander-v2",
    continuous = True,
    render_mode="human"
)

for ep in range(5):
    term = False
    trunc = False
    obs, _ = env.reset()
    while not (term or trunc):
        if obs[0] < -1:
            action = [0,1]
        elif obs[0] > 1:
            action = [2,1]
        elif obs[1] < 0.8 and (not obs[-1] and not obs[-2]):
            action = [1,1]
        else:
            action = [0,0]
        obs, reward, term, trunc, _ = env.step(action)
        env.render()
        print(reward)