import gymnasium as gym
import time
import pygame
import random
from enum import Enum

class PlayMode(Enum):
    MANUAL = 0
    RULE = 1

if __name__ == "__main__":
    pygame.init()
    env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=1000)
    state, _ = env.reset()

    print(f"通关最大步数：{env.spec.max_episode_steps}")

    time.sleep(3)

    start_time = time.time()
    fail = False
    done = False
    step = 0
    play_mode = PlayMode.RULE
    while not done:
        time.sleep(0.3)
        if play_mode == PlayMode.MANUAL:
            # 方式一：键盘输入
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            else:
                action = random.choice([0, 1])
        elif play_mode == PlayMode.RULE:
            # 方式二：简单规则
            # state[0]:cart position(小车位置)
            # state[1]:velocity(小车速度)
            # state[2]:pole angle(杆子角度)
            # state[3]:pole angular velocity(杆子角速度)
            if state[2] <= 0:
                action = 0
            else:
                action = 1

        state, _, terminated, truncated, _ = env.step(action)
        print(f"position={state[0]:.2f}, velocity={state[1]:.2f}, angle={state[2]:.2f}, angle_v={state[3]:.2f}")
        step += 1
        done = terminated or truncated
    
    end_time = time.time()
    game_time = end_time - start_time

    if terminated:
        print(f"game over, play {game_time:.2f} secondes, {step} steps")
    else:
        print(f"success, play {game_time:.2f} secondes, {step} steps")

    time.sleep(10)