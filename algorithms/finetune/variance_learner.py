import torch
from torch import nn
import numpy as np
from iql import MLP
import gym
from typing import Tuple
from matplotlib import pyplot as plt
import os

GAMMA = 0.99

def to_tensor(data: np.ndarray) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)

@torch.no_grad()
def run_episodes(
    env: gym.Env,
    max_steps: int,
    seeds,
    actor=None,
    batch_size=None,
) -> Tuple[np.ndarray, np.ndarray]:
    states = []
    next_states = []
    rewards = []
    dones = []

    if isinstance(seeds, int):
        cond = lambda x: True
        ep = seeds
    else:
        cond = lambda x: x < len(seeds)
        ep = seeds[0]
    
    while cond(ep):
        done = False
        if isinstance(seeds, list) or ep == seeds:
            try:
                env.seed(ep)
                state = env.reset()
            except AttributeError:
                state, _ = env.reset(seed=ep)
                env.action_space.seed(ep)
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
        ts = 0
        while not done:
            states.append(to_tensor(state))
            dones.append(done)
            if actor is None:
                action = env.action_space.sample()
            else:
                action = actor(env, state)
            next_state, reward, next_done, _ = env.step(action)
            real_done = False  # Episode can timeout which is different from done
            if next_done and ts < max_steps:
                real_done = True
            next_done = real_done
            rewards.append(reward)
            next_states.append(to_tensor(next_state))
            if batch_size is not None and len(states) == batch_size:
                return states, rewards, dones, next_states, next_done
            done = next_done
            state = next_state
            ts += 1
        ep += 1
    return states, rewards, dones, next_states, next_done

def nll_loss(pred: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.gaussian_nll_loss(pred, target, var)
    return loss

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean_sq_err = 0.5 * ((pred - target) ** 2).mean()
    return mean_sq_err

class VarianceLearner:
    def __init__(self, state_dim, action_dim, config, actor, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.batch_size = batch_size
        self.actor=actor
        
        self.mf = StateDepFunction(state_dim)
        self.vf = StateDepFunction(state_dim)

        self.m_optimizer = torch.optim.Adam(self.mf.parameters(), lr=.0001)
        self.mv_optimizer = torch.optim.Adam(self.vf.parameters(), lr=.0001)

    def get_values(self, obs, rewards, next_obs, dones, next_done):
        batch_size = len(obs)
        values_samp = torch.zeros(batch_size)
        values_pred = torch.zeros(batch_size)
        variance_pred = torch.zeros(batch_size)
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                nextnonterminal = 1.0 - next_done
                next_val = self.mf(next_obs[t])
            else:
                nextnonterminal = 1.0 - dones[t+1]
                next_val = values_samp[t+1]
            values_pred[t], variance_pred[t] = self.mf(obs[t]), self.vf(obs[t])
            variance_pred[t] = torch.clip(torch.exp(variance_pred[t]), 1e-4, 100000000)
            values_samp[t] = rewards[t] + GAMMA * next_val * nextnonterminal
        return values_samp, values_pred, variance_pred

    def _update_v(self, batch, curr_iter) -> torch.Tensor:
        # Update value function
        (
            observations,
            rewards,
            dones,
            next_observations,
            next_done
        ) = batch

        log_dict = {}
        values_samp, values_pred, variance_pred = self.get_values(observations, rewards, next_observations, dones, next_done)
        v_loss = nll_loss(values_pred, values_samp, variance_pred)

        if curr_iter > 5000:
            self.mv_optimizer.zero_grad()
            v_loss.backward()
            self.mv_optimizer.step()
        else:
            self.m_optimizer.zero_grad()
            v_loss.backward()
            self.m_optimizer.step()

        log_dict["variance_pred_vec"] = variance_pred[:5]
        log_dict["value_loss"] = v_loss.item()
        log_dict["values_pred_vec"] = values_pred[:5]
        log_dict["values_samp_vec"] = values_samp[:5]

        return log_dict

    def run_training(self, env, max_steps, n_updates=10000, evaluate=False):
        vf_losses = []
        log_freq = 10
        for n in range(n_updates):
            batch = run_episodes(env, max_steps, n, batch_size=self.batch_size)
            log_dict = self._update_v(batch, n)
            if n%log_freq == 0:
                print(f"Iteration {n}/{n_updates}:")
                for k,v in log_dict.items():
                    if 'loss' in k:
                        vf_losses.append(v)
                    print(f"{k}: {v}")
        log_dict["var_learner/loss"] = vf_losses
        self.mf.eval()
        self.vf.eval()

        save_path = "jsrl-CORL/algorithms/finetune/var_functions"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if evaluate:
            plt.plot(vf_losses)
            plt.savefig(f"{save_path}/losses_vf.png")
            self.test_model(env, max_steps)
        torch.save(self.vf.state_dict(), f"{save_path}/{env.unwrapped.spec.name}_{self.actor}_{n_updates}.pt")

        del self.mf
        return self.vf
    
    def test_model(self, env, max_steps):
        states, rewards, dones, next_states, next_done = run_episodes(env, max_steps, seeds=[0]*15)
        values_samp, values_pred, variance_pred = self.get_values(states, rewards, next_states, dones, next_done)
        stds_pred = torch.sqrt(variance_pred)
        
        state_keys, val_y_samp, val_y_pred, std_y_pred = {}, {}, {}, {}
        state_key = 0
        for i, state in enumerate(states):
            if state in state_keys:
                state_key = state_keys[state]
                val_y_samp[state_key].append(values_samp[i].detach().numpy().item())
                val_y_pred[state_key].append(values_pred[i].detach().numpy().item())
                std_y_pred[state_key].append(stds_pred[i].detach().numpy().item())
            else:
                state_keys[state] = state_key
                val_y_samp[state_key] = [values_samp[i].detach().numpy().item()]
                val_y_pred[state_key] = [values_pred[i].detach().numpy().item()]
                std_y_pred[state_key] = [stds_pred[i].detach().numpy().item()]
                state_key += 1

        true_y = []
        pred_y = []
        for state_key in state_keys.values():
            for i in range(len(val_y_samp[state_key])):
                true_y.append((state_key, val_y_samp[state_key][i]))
                pred_y.append((state_key, val_y_pred[state_key][i], std_y_pred[state_key][i]))

        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        np.save("jsrl-CORL/algorithms/finetune/true_y.npy", true_y)
        np.save("jsrl-CORL/algorithms/finetune/pred_y.npy", pred_y)  

class StateDepFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
    

if __name__ == "__main__":
    path = "jsrl-CORL/algorithms/finetune/var_functions/antmaze-umaze-diverse_None_10000.pt"

    env = gym.make("antmaze-umaze-diverse-v2")
    max_steps = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    vf = StateDepFunction(state_dim)
    vf = vf.load_state_dict(torch.load(path))
    variance_learner = VarianceLearner(state_dim, action_dim, None, None).run_training(env, max_steps)
    variance_learner.vf = vf

    variance_learner.test_model(env, max_steps)