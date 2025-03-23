from jsrl_utils import make_actor
from jsrl_w_iql import JsrlTrainConfig
from pathlib import Path
import gymnasium
import torch

def evaluate(config: JsrlTrainConfig):
    env = gymnasium.make(config.env, **config.env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    trainer = make_actor(config, state_dim, action_dim, max_action, device=config.device)
    policy_file = Path(config.load_model)
    trainer.load_state_dict(torch.load(policy_file))
    actor = trainer.actor
    i=0
    for ep in config.n_episodes:
        episode_rew = 0
        if i == 0:
            try:
                # gym
                env.seed(config.seed)
                state = env.reset()
            except AttributeError:
                # gymnasium
                state, _ = env.reset(seed=config.seed)
            done = False
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
            done = False
        while not done:
            action = actor.act(state, config.device)
            try:
                state, reward, done, env_infos = env.step(action)
            except ValueError:
                state, reward, term, trunc, env_infos = env.step(action)
                done = term or trunc
            episode_rew += reward
            env.render()
        print(f"Episode {ep} Reward: {episode_rew}")
        