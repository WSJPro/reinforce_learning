"""
Actor-Critic with Truncated GAE 实现
更实用的版本 - 不需要等待episode结束

关键特性:
1. 固定长度的trajectory segments (如每128步更新一次)
2. 使用truncated GAE - 适用于长episode或无限horizon任务
3. 更高的样本效率 - 不浪费未完成的episode
4. 更接近PPO/TRPO的实现方式

与完整GAE的区别:
- 完整GAE: A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t}δ_{T-1}
- Truncated GAE: A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{n-1}δ_{t+n-1} + (γλ)^n V(s_{t+n})
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
    
class ActorCriticTruncatedGAE:
    def __init__(self, actor_lr: float=1e-3, critic_lr: float=1e-3,
                 gamma: float=0.99, lambda_gae: float=0.95,
                 update_steps: int=128, # 每隔多少步更新一次,不需要等episode结束
                 log_dir: str="logs/actor_critic_trucated_gae"):
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
        self.update_steps = update_steps

        self.episode_rewards = []

        self.writer = SummaryWriter(log_dir)
        self.total_steps = 0
        self.current_episode_reward = 0

    def train(self, total_time_steps: int=100000, print_per: int=10):
        state, _ = self.env.reset()
        episode_num = 0

        segment_states = []
        segment_values = []
        segment_log_probs = []
        segment_rewards = []
        segment_dones = []
        
        while self.total_steps < total_time_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                value = self.critic(state_tensor)
            
            probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            # 存储信息
            segment_states.append(state)
            segment_values.append(value)
            segment_log_probs.append(log_prob)
            segment_rewards.append(reward)
            segment_dones.append(done)

            self.current_episode_reward += reward
            self.total_steps += 1

            if done:
                episode_num += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.writer.add_scalar("episode/reward", self.current_episode_reward, episode_num)
                if episode_num % print_per == 0:
                    print(f"[episode {episode_num}] Steps: {self.total_steps}/{total_time_steps} "
                          f"avg_reward {np.mean(self.episode_rewards[-print_per:])}")
                
                self.current_episode_reward = 0
                state, _ = self.env.reset()
            else:
                state = next_state
            
            if len(segment_states) >= self.update_steps or (done and len(segment_states) > 0):
                # 计算最后一个状态的价值(用于truncated GAE)
                if not done:
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        last_value = self.critic(next_state_tensor)
                else:
                    last_value = torch.zeros(1, 1)
                
                # 计算truncated GAE
                advantages = self.compute_truncated_gae(
                    segment_rewards,
                    segment_values,
                    segment_dones,
                    last_value
                )

                # 更新网络
                self.update_networks(
                    segment_states,
                    segment_values,
                    segment_log_probs,
                    advantages
                )   

                # 清空segment
                segment_states = []
                segment_values = []
                segment_log_probs = []
                segment_rewards = []
                segment_dones = []


    def compute_truncated_gae(self, rewards: list[float], values: list[torch.Tensor], dones: list[bool],
                              last_value: torch.Tensor):
        advantages = []
        gae = 0

        values_with_last = values + [last_value]
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            next_value = values_with_last[t + 1].item()

            # 计算TD error
            delta_t = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t].item()

            # 计算GAE
            # 注意：如果done=True, GAE链会被截断
            gae = delta_t + self.lambda_gae * self.gamma * gae * (1 - dones[t])
            advantages.insert(0, gae)
        
        return advantages

    def update_networks(self, states: list[np.ndarray], values: list[torch.Tensor],
                        log_probs: list[torch.Tensor], advantages: list[float]):
        # 转化为tensor
        states_tensor = torch.FloatTensor(np.array(states))
        advantages_tensor = torch.FloatTensor(advantages)
        values_tensor = torch.cat(values)
        # 更新critic网络,必须使用原始advantage
        # 目标：最小化(V(s) - G)^2
        # 其中 G = A(s,a) + V(s) = returns
        returns = advantages_tensor.unsqueeze(-1) + values_tensor.detach()
        predicted_values = self.critic(states_tensor)
        critic_loss = nn.MSELoss()(predicted_values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新策略网络,建议使用标准化后的advantage
        # 目标：最大化 log π(a|s) · A^GAE(s,a)
        if advantages_tensor.shape[0] > 1:
            advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        else:
            advantages_normalized = advantages_tensor
        policy_losses = []
        for log_prob, advantage in zip(log_probs, advantages_normalized):
            policy_losses.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_losses).sum()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

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
    agent = ActorCriticTruncatedGAE(actor_lr=1e-4, critic_lr=1e-3, lambda_gae=0.95, update_steps=128,
                                    log_dir="logs/actor_critic_truncated_gae")
    agent.train()
    agent.evaluate(num_episodes=10)
    agent.save_model(policy_path='pths/ac_truncated_gae_actor.pth',
                     critic_path='pths/ac_truncated_gae_critic.pth')
    play_game(model_path='pths/ac_truncated_gae_actor.pth')





