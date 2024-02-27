import torch
from torch import nn
import numpy as np
from iql import ReplayBuffer, MLP
from jsrl_wrapper import JsrlTrainConfig
import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

GAMMA = 0.9

@torch.no_grad()
def data_collector(
    env: gym.Env,
    max_steps: int,
    config: JsrlTrainConfig,
    rollout_buffer: ReplayBuffer,
    seed = 0,
    actor=None,
) -> Tuple[np.ndarray, np.ndarray]:
    episode_rewards = []
    next_dones = []
    ep = 0
    while True:
        done = False
        if ep == 0:
            try:
                env.seed(seed)
                state = env.reset()
            except AttributeError:
                state, _ = env.reset(seed=seed)
        else:
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
        ts = 0
        episode_reward = 0.0
        
        while not done:
            transition = {"state": state, "action": None,  "reward": None, "next_state": None, "done": done}
            if actor is None:
                transition["action"] = env.action_space.sample()
            else:
                transition["action"] = actor(env, transition["state"])
            transition["next_state"], transition["reward"], next_done, _ = env.step(transition["action"])
            real_done = False  # Episode can timeout which is different from done
            if next_done and ts < max_steps:
                real_done = True
            next_dones.append(real_done)
            rollout_buffer.add_transition(**transition)
            if rollout_buffer._pointer == (rollout_buffer._buffer_size-1):
                return rollout_buffer, next_dones
            
            state = transition["next_state"]
            done = real_done
            episode_reward += transition["reward"]
            ts += 1
            
        episode_rewards.append(episode_reward)
        ep += 1

def mse_loss(pred: torch.Tensor, target: float) -> torch.Tensor:
    #sq_err = (pred - target)**2
    #sum_sq_err = torch.sum(sq_err, axis=1)
    #mean_sq_err = torch.mean(sum_sq_err, axis=-1)
    mean_sq_err = 0.5 * ((pred - target) ** 2).mean()
    return mean_sq_err

class VarianceLearner:
    def __init__(self, state_dim, action_dim, config, actor, batch_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.batch_size = batch_size
        self.actor=actor
        
        self.vf = ValueFunction(state_dim)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=.00005)


        self.varf = ValueFunction(state_dim)
        self.var_optimizer = torch.optim.Adam(self.varf.parameters(), lr=.0001)
        

    def _update_v(self, batch, next_dones, n) -> torch.Tensor:
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
                nextnonterminal = 1.0 - next_dones[t-1]
                next_val = self.vf(next_observations[t-1]).clone().detach()
            else:
                nextnonterminal = 1.0 - dones[t+1]
                next_val = values_samp[t+1]
            values_pred[t] = self.vf(observations[t])
            variance_pred[t] = self.varf(observations[t])
            values_samp[t] = rewards[t] + GAMMA * next_val * nextnonterminal
            variance_samp[t] = (values_samp[t]-values_pred[t].clone().detach())**2
           
        #preds = torch.stack((values_pred, variance_pred), axis=-1)
        #samps = torch.stack((values_samp, variance_samp), axis=-1)
        
        v_loss = mse_loss(values_pred, values_samp)
        var_loss = mse_loss(variance_pred, variance_samp)

        log_dict["variance_pred_vec"] = {variance_pred[:5]}
        log_dict["variance_samp_vec"] = {variance_samp[:5]}
        
        log_dict["value_loss"] = v_loss.item()
        log_dict["var_loss"] = var_loss.item()

        log_dict["values_pred_vec"] = {values_pred[:5]}
        log_dict["values_samp_vec"] = {values_samp[:5]}
        

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()


        if (n+1) % 10:
            self.var_optimizer.zero_grad()
            var_loss.backward()
            self.var_optimizer.step()
        
        #self.v_optimizer.zero_grad()
        #v_loss.backward()
        #self.v_optimizer.step()

        

        return log_dict

    def run_training(self, env, max_steps, n_updates=10000):
        rollout_buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.batch_size,
            self.config.device,
        )
        
        var_losses = []
        vf_losses = []
        for n in range(n_updates):
            updated_buffer, next_dones = data_collector(env, max_steps, self.config, rollout_buffer, seed=n, actor=self.actor)
            batch = updated_buffer.sample(self.batch_size)
            batch = [b.to(self.config.device) for b in batch]
            log_dict = self._update_v(batch, next_dones, n)
            if n%5 == 0:
                print(f"Iteration {n}/{n_updates}:")
                for k,v in log_dict.items():
                    if k=="var_loss":
                        print(f"{k}: {np.format_float_scientific(v)}")
                        var_losses.append(v)
                    elif k =="value_loss":
                        vf_losses.append(v)
                    else:
                        print(f"{k}: {v}")
        
        import matplotlib.pyplot as plt
        plt.plot(var_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Variance Loss")
        plt.savefig("jsrl-CORL/algorithms/finetune/losses_var.png")
        plt.close()
        plt.plot(vf_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Value Loss")
        plt.savefig("jsrl-CORL/algorithms/finetune/losses_vf.png")
     
        self.vf.eval()
        
        return self.vf

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
    