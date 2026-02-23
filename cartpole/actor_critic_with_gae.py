"""
Actor-Critic with GAE (Generalized Advantage Estimation) 算法实现
基于actor_critic.py改造

GAE的核心思想:
- 通过λ参数在偏差(bias)和方差(variance)之间进行权衡
- λ=0: 等价于TD(0), 低方差但高偏差
- λ=1: 等价于Monte Carlo, 低偏差但高方差
- 0<λ<1: 平衡偏差和方差

关键修改点:
1. 训练方式: 从step-level改回episode-level (需要收集完整轨迹)
2. Advantage计算: 从TD error改为GAE
   - TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
   - GAE: A^GAE_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
3. 更新时机: 在episode结束后批量更新

GAE公式推导:
A^GAE(γ,λ)_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
             = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
"""

import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter

class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)
    
class ActorCriticWithGAE:
    def __init__(self, actor_lr: float=1e-3, critic_lr: float=1e-3,
                 gamma: float=0.99, lambda_gae: float=0.95,
                 log_dir: str="logs/actor_critic_with_gae"):
        self.env = gym.make("CartPole-v1", max_episode_steps=1000)
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 128

        self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.episode_rewards = []

        self.writer = SummaryWriter(log_dir)

    def train(self, num_episodes: int=1000, print_per: int=10):
        for episode in range(1, num_episodes + 1):
            episode_reward = self.train_episode()
            self.episode_rewards.append(episode_reward)
            self.writer.add_scalar("episode/reward", episode_reward, episode)
            if episode % print_per == 0:
                print(f"[{episode}/{num_episodes}] avg_reward {np.mean(self.episode_rewards[-print_per:])}")

    def train_episode(self):
        episode_reward = 0
        state, _ = self.env.reset()
        states = []
        values = []
        rewards = []
        log_probs = []
        done = False

        is_terminated = False
        # 1. 收集episode轨迹
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            value = self.critic(state_tensor)
            probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            episode_reward += reward
            done = terminated or truncated
            if done:
                is_terminated = terminated

            states.append(state)
            values.append(value)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
        
        # 2. 计算广义优势函数GAE
        advantages = self.compute_gae(values, rewards, state, is_terminated)

        # 3. 更新网络
        self.update_networks(states, values, advantages, log_probs, rewards, state,
                             is_terminated)

        return episode_reward
        
    def compute_gae(self, values: list[torch.Tensor], rewards: list[float],
                    final_state: np.ndarray, is_terminated: bool):
        """
        计算Generalized Advantage Estimation (GAE)
        
        GAE公式:
        A^GAE_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """

        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                if is_terminated:
                    next_value = 0
                else:
                    final_state_tensor = torch.FloatTensor(final_state).unsqueeze(0)
                    with torch.no_grad():
                        next_value = self.critic(final_state_tensor).item()
            else:
                next_value = values[t + 1].item()
            # 计算TD error
            delta_t = rewards[t] + self.gamma * next_value - values[t].item()
            # 计算GAE: A_t = δ_t + (γλ)A_{t+1}
            gae = delta_t + self.gamma * self.lambda_gae * gae

            advantages.insert(0, gae)
        
        return advantages

    def update_networks(self, states: list[np.ndarray], values: list[torch.Tensor],
                        advantages: list[float], log_probs: list[torch.Tensor],
                        rewards: list[float], final_state: np.ndarray, is_terminated: bool):
        states_tensor = torch.FloatTensor(np.array(states))
        advantages_tensor = torch.FloatTensor(advantages)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        values_tensor = torch.cat(values).squeeze()
        advantages_raw = torch.FloatTensor(advantages)
        # 方法1: 使用GAE的returns: G_t = A_t + V(s_t)
        returns = advantages_raw + values_tensor.detach()
        # 方法2: 也可以使用discounted returns
        # returns = self.compute_returns(rewards, final_state, is_terminated)
        
        # 1. 更新critic
        # 目标：最小化 (V(s) - G_t)^2
        predicted_values = self.critic(states_tensor).squeeze()
        critic_loss = nn.MSELoss()(returns, predicted_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. 更新Actor
        # 目标：最大化 log π(a|s) · A^GAE(s,a)
        policy_losses = []
        for log_prob, advantage in zip(log_probs, advantages_tensor):
            policy_losses.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_losses).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
    
    def compute_returns(self, rewards: list[float], final_state: np.ndarray,
                        is_terminated: bool):
        """
        计算折扣回报 (用于对比)
        G_t = r_t + γr_{t+1} + γ^2r_{t+2} + ...
        """
        
        returns = []
        if is_terminated:
            G = 0
        else:
            final_state_tensor = torch.FloatTensor(final_state).unsqueeze(0)
            with torch.no_grad():
                G = self.critic(final_state_tensor).item()
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return torch.FloatTensor(returns)


    def evaluate(self, num_episodes: int = 10):
        episode_rewards = []
        for i in range(num_episodes):
            if i == 0:
                state, _ = self.env.reset(seed=42)  # 只在第一次设置种子
            else:
                state, _ = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    probs = self.actor(state_tensor)
                action = torch.argmax(probs, dim=-1).item()
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        print(f"evaluate result: avg_reward = {avg_reward:.2f}")
    
    def save_model(self, policy_path: str, critic_path: str):
        # 创建策略网络的目录
        policy_dir = os.path.dirname(policy_path)
        if policy_dir and not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
            print(f"创建目录：{policy_dir}")
        
        # 创建价值网络的目录
        critic_dir = os.path.dirname(critic_path)
        if critic_dir and not os.path.exists(critic_dir):
            os.makedirs(critic_dir)
            print(f"创建目录：{critic_dir}")
        torch.save(self.actor.state_dict(), policy_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"策略网络已保存：{policy_path}")
        print(f"价值网络已保存：{critic_path}")


def play_game(model_path: str):
    env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=1000)
    state, _ = env.reset()

    print(f"通关最大步数：{env.spec.max_episode_steps}")

    time.sleep(3)

    policy = ActorNetwork()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    start_time = time.time()
    done = False
    step = 0
    while not done:
        time.sleep(0.01)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = policy(state_tensor)
        action = torch.argmax(probs, -1).item()

        state, _, terminated, truncated, _ = env.step(action)
        step += 1
        done = terminated or truncated
    
    end_time = time.time()
    game_time = end_time - start_time

    if terminated:
        print(f"game over, play {game_time:.2f} secondes, {step} steps")
    else:
        print(f"success, play {game_time:.2f} secondes, {step} steps")

    time.sleep(3)


if __name__ == "__main__":
    agent = ActorCriticWithGAE(actor_lr=1e-4, critic_lr=1e-3)
    agent.train(num_episodes=1000, print_per=10)
    agent.evaluate(num_episodes=10)
    agent.save_model(policy_path='pths/ac_with_gae_actor.pth',
                     critic_path='pths/ac_with_gae_critic.pth')
    play_game(model_path='pths/ac_with_gae_actor.pth')





