import torch
from torch import nn
import numpy as np
from iql import ReplayBuffer, MLP
from jsrl_wrapper import JsrlTrainConfig
import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

GAMMA = 0.99
LR = 0.001

@torch.no_grad()
def data_collector(
    env: gym.Env,
    max_steps: int,
    config: JsrlTrainConfig,
    rollout_buffer: ReplayBuffer
) -> Tuple[np.ndarray, np.ndarray]:

    episode_rewards = []
    get_next_done = False
    for i in range(config.n_episodes):
        done = False
        if i == 0:
            try:
                env.seed(config.seed)
                state = env.reset()
            except AttributeError:
                state, _ = env.reset(seed=config.seed)
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
        ts = 0
        episode_reward = 0.0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            real_done = False  # Episode can timeout which is different from done
            if done and ts < max_steps:
                real_done = True
            if get_next_done:
                return rollout_buffer, real_done
            if rollout_buffer._pointer == (rollout_buffer._buffer_size-1):
                get_next_done = True
            rollout_buffer.add_transition(
                state, action, reward, next_state, real_done
            )
            state = next_state
            episode_reward += reward
            ts += 1
            
        episode_rewards.append(episode_reward)

def mse_loss(pred: torch.Tensor, target: float) -> torch.Tensor:
    return torch.mean(torch.mean(torch.sum((pred - target)**2), axis=-1), axis=0)

class VarianceLearner:
    def __init__(self, state_dim, action_dim, config, batch_size=256):
        self.vf = ValueFunction(state_dim)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=LR)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.batch_size = batch_size

    def _update_v(self, batch, next_done) -> torch.Tensor:
        # Update value function
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones
        ) = batch

        log_dict = {}

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
                              
        values_samp = torch.zeros(self.batch_size)
        values_pred = torch.zeros(self.batch_size)
        variance_samp = torch.zeros(self.batch_size)
        variance_pred = torch.zeros(self.batch_size)
        for t in reversed(range(self.batch_size)):
            if t == self.batch_size - 1:
                nextnonterminal = 1.0 - next_done
                next_val = 0
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_val = 0
            #import pdb;pdb.set_trace()
            values_pred[t], variance_pred[t] = self.vf(observations[t])
            values_samp[t] = rewards[t] + GAMMA * next_val * nextnonterminal
            variance_samp[t] = (values_samp[t]-values_pred[t])**2
            next_val = values_samp[t]

        preds = np.array(list(zip(values_pred, variance_pred)))
        samps = np.array(list(zip(values_samp, variance_samp)))

        v_loss = mse_loss(preds, samps)
        
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return log_dict

    def run_training(self, env, max_steps, n_updates=100):
        rollout_buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.batch_size,
            self.config.device,
        )

        for n in range(n_updates):
            updated_buffer, next_done = data_collector(env, max_steps, self.config, rollout_buffer)
            batch = updated_buffer.sample(self.batch_size)
            batch = [b.to(self.config.device) for b in batch]
            log_dict = self._update_v(batch, next_done)
            print(f"Iteration {n}/{n_updates}: {log_dict}")
        self.vf.eval()
        return self.vf

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 2]
        self.v = MLP(dims, squeeze_output=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
