from gymnasium.core import Env
from gymnasium.spaces import Box, Discrete
import gymnasium
import torch
from torch import nn
from torch.distributions import Distribution
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Tuple
from torch.optim import Adam
import numpy as np

class Traj(object):
    
    def __init__(self):
        
        self._obs = []
        self._act = []
        self._reward = []
        self._value = []
    
    def append(self, obs, act, reward, value):
        
        self._obs.append(obs)
        self._value.append(value)
        self._act.append(act)
        self._reward.append(reward)
    
    def finish_traj(self, gamma):
        # compute reward-to-go (rtg)
        n = len(self._reward)
        reward = np.array(self._reward).reshape(-1)
        discounts = gamma ** np.arange(n, dtype=np.float32)
        rtg = np.flip(np.cumsum(np.flip(reward * discounts, axis=(0,)), axis=0), axis=(0,))
        rtg /= discounts
        
        # compute advantage
        value = np.array(self._value).reshape(-1)
        adv = rtg - value
        
        return (self._obs, self._act, rtg, adv)

class TrajBuffer(object):
    
    def __init__(self):
        
        self._obs, self._act, self._rtg, self._adv = [], [], [], []
    
    def reset(self):
        
        self._obs.clear()
        self._act.clear()
        self._rtg.clear()
        self._adv.clear()
    
    def append(self, obs, act, rtg, adv):
        
        self._obs.append(obs)
        self._act.append(act)
        self._rtg.append(rtg)
        self._adv.append(adv)
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return (
            torch.as_tensor(np.concatenate(self._obs, axis=0)),
            torch.as_tensor(np.concatenate(self._act, axis=0)),
            torch.as_tensor(np.concatenate(self._rtg, axis=0)),
            torch.as_tensor(np.concatenate(self._adv, axis=0)),
            )

class Actor(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def _distribution(self, obs):
        """
        This function should return a distribution given observation \pi_{\theta} ( \dot | s )
        """
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi: Distribution, act):
        """
        This function gives the log likelihood of a given act under a given distribution (the distribution is already conditioned by observation)
        """
        raise NotImplementedError
    
    def forward(self, obs, act):
        """
        Gives the log likelihood of the given action conditioned on the observation
        """
        pi = self._distribution(obs)
        log_prob = self._log_prob_from_distribution(pi, act)
        return log_prob
    
    def step(self, obs):
        with torch.no_grad():
            pi = self._distribution(obs)
            act = pi.sample()
        return act

class CategoricalActor(Actor):

    def __init__(self, act_net: nn.Module):
        super().__init__()
        self._logits_net = act_net
    
    def _distribution(self, obs):
        return Categorical(logits=self._logits_net(obs))
    
    def _log_prob_from_distribution(self, pi: Distribution, act):
        return pi.log_prob(act)

class GaussianActor(Actor):
    
    def __init__(self, act_dim: int, log_std: float, act_net: nn.Module):
        super().__init__()
        _log_std = log_std * torch.ones(act_dim, dtype=torch.float32)
        self._log_std = torch.nn.Parameter(_log_std)
        self._mu_net = act_net

    def _distribution(self, obs):
        return Normal(self._mu_net(obs), torch.exp(self._log_std))
    
    def _log_prob_from_distribution(self, pi: Distribution, act):
        # The sum equals to probability product, which give the joint probability of multi-dimension independent distributions
        return pi.log_prob(act).sum(axis=-1)

class Critic(nn.Module):
    
    def __init__(self, critic_net: nn.Module):
        super().__init__()
        self._critic_net = critic_net
    
    def forward(self, obs):
        """
        when grad is required
        """
        return torch.squeeze(self._critic_net(obs), -1)

    def step(self, obs):
        """
        when no grad is needed
        """
        with torch.no_grad():
            value = torch.squeeze(self._critic_net(obs), -1)
        return value

class VPG(object):
    
    def __init__(self, env: Env, log_std: float, act_net: nn.Module, critic_net: nn.Module, policy_lr: float, value_lr: float) -> None:
        
        self._env = env
        
        if isinstance(self._env.action_space, Box):
            self._policy_net = GaussianActor(env.action_space.shape[0], log_std, act_net)
        elif isinstance(self._env.action_space, Discrete):
            self._policy_net = CategoricalActor(act_net)
        else:
            raise NotImplementedError
        
        self._value_net = Critic(critic_net)
        
        self._traj_buffer = TrajBuffer()
        
        self._policy_optimizer = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._value_optimizer = Adam(self._value_net.parameters(), lr=value_lr)
    
    def train(
        self, 
        epochs: int, 
        value_iters: int,
        traj_num_per_epoch: int, 
        max_step: int,
        gamma: float
    ):
        for k in range(epochs):
            
            epoch_reward = 0.0
            
            self._traj_buffer.reset()
            
            for i in range(traj_num_per_epoch):
                
                traj = Traj()
                
                obs, _ = self._env.reset()
                done = False
                
                time_step = 0
                traj_reward = 0.0
                
                while not done and time_step < max_step:
                    act = self._policy_net.step(torch.tensor(obs, dtype=torch.float32))
                    value = self._value_net.step(torch.tensor(obs, dtype=torch.float32))
                    new_obs, reward, terminated, truncated, _ = self._env.step(act.numpy())
                    
                    done = terminated or truncated
                    
                    traj.append(obs, act, reward, value.numpy())
                    
                    obs = new_obs
                    
                    time_step += 1
                    
                    traj_reward += reward
                
                epoch_reward += traj_reward
                
                self._traj_buffer.append(*traj.finish_traj(gamma))
            
            obs, act, rtg, adv = self._traj_buffer.get()
            
            # optimize policy network
            self._policy_optimizer.zero_grad()
            log_prob = self._policy_net(obs, act)
            policy_loss = - (log_prob * adv).mean()
            policy_loss.backward()
            self._policy_optimizer.step()
            
            # optimize value network
            for _ in range(value_iters):
                self._value_optimizer.zero_grad()
                value = self._value_net(obs)
                value_loss = ((value - rtg) ** 2).mean()
                value_loss.backward()
                self._value_optimizer.step()
            
            print("reward:", epoch_reward / traj_num_per_epoch, "progress:", k / epochs * 100.0, "%")
        
        self._env.close()

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    
    vpg = VPG(
        env=gymnasium.make('CartPole-v1'),
        log_std=0.5,
        act_net=MLP(4, 20, 2),
        critic_net=MLP(4, 20, 1),
        policy_lr=0.01,
        value_lr=0.01
        )
    
    vpg.train(100, 100, 50, 1000, 0.99)