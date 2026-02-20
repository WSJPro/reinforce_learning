import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Optional

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCE:
    def __init__(
            self, 
            max_episode_steps: int = 1000, 
            gamma: float = 0.99,
            learning_rate: float = 1e-3,
            log_dir: str = "logs/reinforce",
            seed: Optional[int] = None
            ):
        self.env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps, )
        self.gamma = gamma
        self.episode_rewards = []
        self.final_step_records = []

        if seed is not None:
            self.env.reset(seed=seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        state_dim = self.env.observation_space.shape[0]  # 4
        action_dim = self.env.action_space.n             # 2
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(log_dir)


    def train(self, num_episodes: int = 1000, print_per: int = 10):
        for episode in range(1, num_episodes + 1):
            episode_reward, step, loss = self.train_episode()
            self.writer.add_scalar("episode/reward", episode_reward, episode)
            self.writer.add_scalar("episode/loss", loss, episode)
            self.episode_rewards.append(episode_reward)
            self.final_step_records.append(step)
            if episode % print_per == 0:
                print(f"episode_num = {episode}, average_step = {np.mean(self.final_step_records[-print_per:])}, "
                      f"average_reward = {np.mean(self.episode_rewards[-print_per:])}")
        
        print("训练结束")

    def train_episode(self):
        state, _ = self.env.reset()

        log_probs = []
        rewards = []
        step = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            step += 1
            done = terminated or truncated
        
        returns = self.compute_returns(rewards)
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.sum(rewards), step, loss.item()
    
    def compute_returns(self, rewards: list[float])->list[float]:
        # 计算折扣回报 G_t = r_t + gamma * r_{t+1} + gamma^2*r_{t+2}+...
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

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
                    probs = self.policy(state_tensor)
                action = torch.argmax(probs, -1).item()
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        print(f"avg_reward = {avg_reward:.2f}")

    def save_model(self, model_path: str = 'cartpole_policy_reinforce.pth'):
        torch.save(self.policy.state_dict(), model_path)
        print(f"模型已保存：{model_path}")



def play_game(model_path: str):
    env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=1000)
    state, _ = env.reset()

    print(f"通关最大步数：{env.spec.max_episode_steps}")

    time.sleep(3)

    policy = PolicyNetwork()
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
        # print(f"step={step}, position={state[0]:.2f}, velocity={state[1]:.2f}, angle={state[2]:.2f}, angle_v={state[3]:.2f}")
        done = terminated or truncated
    
    end_time = time.time()
    game_time = end_time - start_time

    if terminated:
        print(f"game over, play {game_time:.2f} secondes, {step} steps")
    else:
        print(f"success, play {game_time:.2f} secondes, {step} steps")

    time.sleep(3)

if __name__ == "__main__":
    agent = REINFORCE(learning_rate=1e-4, log_dir="logs/reinforce")
    agent.train(num_episodes=1000, print_per=10)
    agent.evaluate()
    agent.save_model(model_path = 'cartpole_policy_reinforce.pth')
    play_game(model_path = 'cartpole_policy_reinforce.pth')

