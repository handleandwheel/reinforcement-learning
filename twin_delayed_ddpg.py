from gymnasium.core import Env
from gymnasium.spaces import Box, Discrete
import gymnasium
import torch
from torch import nn
from torch.distributions import Distribution
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import Tuple, List
from torch.optim import Adam
import torch.autograd as autograd
import numpy as np
import itertools

class ReplayBuffer(object):
    
    def __init__(self, size: int, reward_scale: float, obs_dim: int, act_dim: int):
        
        self._buffer_size = size
        self._reward_scale = reward_scale
        
        self._obs = np.zeros([self._buffer_size, obs_dim], dtype=np.float32)
        self._act = np.zeros([self._buffer_size, act_dim], dtype=np.float32)
        self._next_obs = np.zeros([self._buffer_size, obs_dim], dtype=np.float32)
        self._rew = np.zeros(self._buffer_size, dtype=np.float32)
        self._done = np.zeros(self._buffer_size, dtype=np.float32)
        
        self._ptr = 0
        self._full = False
    
    def sample(self, batch: int):
        
        if self._full:
            inds = np.random.randint(0, self._buffer_size, batch)
        else:
            inds = np.random.randint(0, self._ptr, batch)

        return (
            torch.tensor(self._obs[inds]),
            torch.tensor(self._act[inds]),
            torch.tensor(self._rew[inds]),
            torch.tensor(self._next_obs[inds]),
            torch.tensor(self._done[inds]),
        )
    
    def append(self, obs, act, rew, next_obs, done):
        
        self._obs[self._ptr] = obs
        self._act[self._ptr] = act
        self._rew[self._ptr] = self._reward_scale * rew
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr] = done
        
        self._ptr += 1
        
        if self._ptr == self._buffer_size:
            self._ptr = 0
            self._full = True

    def reset(self):
        
        self._ptr = 0
        self._full = False

class ActorMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size: List[int], output_size):
        
        super(ActorMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden_size[2], output_size)
        self.relu4 = nn.Tanh()
        
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        
        return x

class CriticMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size: List[int], output_size):
        
        super(CriticMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size[2], output_size)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        
        return x

class Actor(nn.Module):
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: List[int]):
        super().__init__()
        self._act_net = ActorMLP(obs_dim, hidden_size, act_dim)
    
    def forward(self, obs):
        return self._act_net(obs)

class Critic(nn.Module):
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: List[int]):
        super().__init__()
        self._critic_net = CriticMLP(obs_dim + act_dim, hidden_size, 1)
    
    def forward(self, obs, act):
        q = self._critic_net(torch.cat((obs, act), dim=1))
        return torch.squeeze(q, -1)

class TD3(object):
    
    def __init__(self, env: Env, actor_hidden: List[int], critic_hidden: List[int], policy_lr: float, value_lr: float, buffer_size: int, reward_scale: float, device: str) -> None:
        """
        the action and observation should both be normalized to [-1, 1]
        """
        self._env_fn = env
        
        self._env = self._env_fn(render = False)
        
        assert isinstance(self._env.action_space, gymnasium.spaces.Box)
        
        self._obs_dim = self._env.observation_space.shape[0]
        self._act_dim = self._env.action_space.shape[0]
        
        self._mu_net = Actor(self._obs_dim, self._act_dim, actor_hidden)
        self._q1_net = Critic(self._obs_dim, self._act_dim, critic_hidden)
        self._q2_net = Critic(self._obs_dim, self._act_dim, critic_hidden)
        
        self._mu_target = Actor(self._obs_dim, self._act_dim, actor_hidden)
        self._q1_target = Critic(self._obs_dim, self._act_dim, critic_hidden)
        self._q2_target = Critic(self._obs_dim, self._act_dim, critic_hidden)
        
        self._mu_target.load_state_dict(self._mu_net.state_dict())
        self._q1_target.load_state_dict(self._q1_net.state_dict())
        self._q2_target.load_state_dict(self._q2_net.state_dict())
        
        self._replay_buffer = ReplayBuffer(int(buffer_size), reward_scale, self._obs_dim, self._act_dim)
        
        self._mu_optimizer = Adam(self._mu_net.parameters(), lr=policy_lr)
        self._q_optimizer = Adam(itertools.chain(self._q1_net.parameters(), self._q2_net.parameters()), lr=value_lr)
        
        self._device = device
        
    
    def eval(
        self,
        epochs: int,
        max_step: int
    ) -> None:
        
        eval_env_multiple = self._env_fn(render = False)
        eval_env_single = self._env_fn(render = True)
        
        obs, _ = eval_env_single.reset()
            
        done = False
        
        time_step = 0
        
        while not done and time_step < max_step:
            with torch.no_grad():
                act = self._mu_net(torch.tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = eval_env_single.step(act.numpy())
            
            done = terminated or truncated
            
            time_step += 1
        
        epoch_reward = 0.0
        
        for i in range(epochs):
            
            obs, _ = eval_env_multiple.reset()
            
            done = False
            
            time_step = 0
            traj_reward = 0.0
            
            while not done and time_step < max_step:
                with torch.no_grad():
                    act = self._mu_net(torch.tensor(obs, dtype=torch.float32))
                obs, reward, terminated, truncated, _ = eval_env_multiple.step(act.numpy())
                
                done = terminated or truncated
                
                time_step += 1
                
                traj_reward += reward
            
            epoch_reward += traj_reward
        
        print("reward:", epoch_reward / epochs)

        eval_env_multiple.close()
        eval_env_single.close()
    
    def train(
        self, 
        epochs: int = 10000,
        update_period: int = 10, # how many epoch per update round
        update_num: int = 1000, # how many update per update round
        batch_size: int = 128,
        start_after: int = 1000, # how many epoch before starting training
        warm_up_epochs: int = 100,
        rho = 0.995,
        explore_sigma: float = 0.3,
        action_noise_bound: float = 0.45,
        gamma: float = 0.99,
        policy_delay: int = 2,
        eval_freq: int = 200,
        max_step: int = 200
    ):
        self._replay_buffer.reset()
        
        for k in range(epochs):
            
            obs, _ = self._env.reset()
            done = False
            time_step = 0
            while not done:
                if k >= warm_up_epochs:
                    with torch.no_grad():
                        act = torch.clamp(self._mu_net(torch.tensor(obs, dtype=torch.float32)) + np.random.randn(self._act_dim) * explore_sigma, -1.0, 1.0)
                        act = act.numpy()
                else:
                    act = (np.random.rand(self._act_dim) - 0.5) * 2.0
                    
                next_obs, reward, terminated, truncated, _ = self._env.step(act)
                done = terminated or truncated or time_step >= max_step
                self._replay_buffer.append(obs, act, reward, next_obs, done)
                obs = next_obs
                time_step += 1
            
            if (k + 1) % update_period == 0 and k > start_after:
                
                for i in range(update_num):
                
                    obss, acts, rews, next_obss, dones = self._replay_buffer.sample(batch_size)
                    
                    with torch.no_grad():
                    
                        next_acts = torch.clamp(
                            self._mu_target(next_obss) \
                            + torch.clamp(torch.randn(self._act_dim) * explore_sigma, - action_noise_bound, action_noise_bound), -1.0, 1.0)
                        
                        q1 = self._q1_target(next_obss, next_acts)
                        q2 = self._q2_target(next_obss, next_acts)
                        targets = rews + gamma * (1 - dones) * torch.min(q1, q2)

                    # optimize q
                    self._q_optimizer.zero_grad()
                    q1_loss = ((self._q1_net(obss, acts) - targets) ** 2).mean()
                    q2_loss = ((self._q2_net(obss, acts) - targets) ** 2).mean()
                    q_loss = q1_loss + q2_loss
                    q_loss.backward()
                    self._q_optimizer.step()
                    
                    if (i + 1) % policy_delay == 0:
                    
                        # optimize mu
                        for q in self._q1_net.parameters():
                            q.requires_grad = False
                            
                        self._mu_optimizer.zero_grad()
                        mu_loss = - self._q1_net(obss, self._mu_net(obss)).mean()
                        mu_loss.backward()
                        self._mu_optimizer.step()
                    
                        for q in self._q1_net.parameters():
                            q.requires_grad = True
                    
                        with torch.no_grad():
                            for mu_param, target_param in zip(self._mu_net.parameters(), self._mu_target.parameters()):
                                target_param.data.mul_(rho)
                                target_param.data.add_((1 - rho) * mu_param.data)
                                
                            for q1_param, target_param in zip(self._q1_net.parameters(), self._q1_target.parameters()):
                                target_param.data.mul_(rho)
                                target_param.data.add_((1 - rho) * q1_param.data)
                                
                            for q2_param, target_param in zip(self._q2_net.parameters(), self._q2_target.parameters()):
                                target_param.data.mul_(rho)
                                target_param.data.add_((1 - rho) * q2_param.data)
            
            # # optimization
            # if not self._device == 'cpu':
            #     self._policy_net.to(self._device)
            #     self._value_net.to(self._device)
            #     obs = obs.to(self._device)
            #     act = act.to(self._device)
            #     rtg = rtg.to(self._device)
            #     adv = adv.to(self._device)
            #     old_log_prob = old_log_prob.to(self._device)
            
            # # optimize policy network
            # self._policy_optimizer.zero_grad()
            # log_prob = self._policy_net(obs, act)
            # ratio = torch.exp(log_prob - old_log_prob)
            # clipped_adv = torch.clamp(ratio, 1-self._clip_thresh, 1+self._clip_thresh) * adv
            # policy_loss = - torch.min(clipped_adv, ratio * adv).mean()
            # policy_loss.backward()
            # self._policy_optimizer.step()
            
            # # optimize value network
            # for _ in range(value_iters):
            #     self._value_optimizer.zero_grad()
            #     value = self._value_net(obs)
            #     value_loss = ((value - rtg) ** 2).mean()
            #     value_loss.backward()
            #     self._value_optimizer.step()
            
            # if not self._device == 'cpu':
            #     self._policy_net.to('cpu')
            #     self._value_net.to('cpu')
            
            # print("reward:", epoch_reward / traj_num_per_epoch, "progress:", k / epochs * 100.0, "%")

            if (k + 1) % eval_freq == 0 and k >= start_after:
                self.eval(15, max_step)
            
            if k % (epochs // 100) == 0:
                print("progress:", k / (epochs // 100), "%")
        
        self._env.close()

class EnvWrapper(object):
    
    def __init__(self, render: bool = False):
        self._env = gymnasium.make("Pendulum-v1", render_mode=("human" if render else None))
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
    def step(self, action):
        action = 1.0 * action
        return self._env.step(action)
    
    def reset(self):
        return self._env.reset()
    
    def close(self):
        return self._env.close()

if __name__ == "__main__":
    
    # the parameters are extremely hard to tune
    
    td3 = TD3(
        env=EnvWrapper,
        actor_hidden=[40, 40, 40],
        critic_hidden=[40, 40, 40],
        policy_lr=0.01,
        value_lr=0.01,
        buffer_size=5e5,
        reward_scale=0.01,
        device='cpu'
        )
    
    td3.train()