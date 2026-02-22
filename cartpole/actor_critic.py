"""
Actor-Critic 算法实现
从REINFORCE with Baseline 改造而来

关键修改点:
1. 训练方式: 从episode-level 改为 step-level
2. Adavantage计算: 从Monte Carlo回报(Gt - V(s)) 改为 TD error (r + gamma * V(s') - V(s))
3. 更新时机: 从收集完整轨迹后批量更新,改为每一步立即更新
"""
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os

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

class ActorCritic:
    def __init__(
            self,
            actor_lr: float = 1e-3,
            critic_lr: float = 1e-3,
            gamma: float = 0.99,
            log_dir: str = "logs/actor_critic"):
        self.env = gym.make("CartPole-v1", max_episode_steps=1000)
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_dim = 128

        self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticNetwork(state_dim=self.state_dim, hidden_dim=self.hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma

        self.episode_rewards = []

        self.writer = SummaryWriter(log_dir)

    def train(self, num_episodes: int = 1000, print_per: int = 10):
        for episode in range(1, num_episodes + 1):
            episode_reward = self.train_episode()
            self.episode_rewards.append(episode_reward)
            self.writer.add_scalar("episode/reward", episode_reward, episode)
            if episode % print_per == 0:
                print(f"[{episode}/{num_episodes}] avg_reward {np.mean(self.episode_rewards[-print_per:])}")
        

    def train_episode(self):
        state, _ = self.env.reset()

        episode_reward = 0
        done = False
        while not done:
            # 1. 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # 2. 执行动作，获取下一个状态和奖励
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

            # 3. 更新价值网络
            # REINFORCE需要等到episode结束才更新，Actor-Critic每步都更新
            critic_loss, policy_loss = self.update_network(state, log_prob, reward, next_state, done)

            state = next_state

        return episode_reward

    def update_network(self, state: np.ndarray, log_prob: torch.Tensor, reward: float, next_state: np.ndarray, done: bool):
        """同时更新Actor和critic网络"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # 1. 计算当前状态和下一个状态的价值 
        state_value = self.critic(state_tensor) # 需要梯度，用于critic更新
        with torch.no_grad():
            if not done:
                next_state_value = self.critic(next_state_tensor) # 不需要梯度
            else:
                next_state_value = torch.zeros_like(state_value)

        # 核心公式: TD error (Temporal Difference Error)
        # δ_t = r_t + γV(s_{t+1}) - V(s_t)
        td_target = reward + self.gamma * next_state_value
        td_error = td_target - state_value

        # 2. 更新critic网络
        # 目标：最小化 TD error 的平方
        # Loss = (δ_t)^2 = (r + γV(s') - V(s))^2
        critic_loss = torch.square(td_error)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. 更新策略网络
        # 使用TD error作为advantage
        # Loss = -log π(a|s) · δ_t
        advantage = td_error.detach() # 不对advantage进行梯度传播
        policy_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), policy_loss.item()

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
                state_tensor = torch.FloatTensor(state)
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
        time.sleep(0.3)
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
    agent = ActorCritic(actor_lr=1e-4, critic_lr=1e-3)
    agent.train(num_episodes=1000, print_per=10)
    agent.evaluate(num_episodes=10)
    agent.save_model(policy_path='pths/cartpole_policy_actor_critic.pth', critic_path='pths/cartpole_critic_actor_critic.pth')
    play_game(model_path='pths/cartpole_policy_actor_critic.pth')

