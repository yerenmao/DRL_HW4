import os
import sys
import time

import torch
import numpy as np
import gymnasium as gym

from core.agent import Agent
from core.config import TOTAL_EPISODES, EPISODE_MAX_STEP, SEED
from plot import plot_rewards

sys.path.append(os.path.abspath(".."))
from dmc import make_dmc_env

def train(num_episodes, checkpoint_dir):
    env = make_dmc_env("cartpole-balance", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    agent = Agent()

    history = []
    start_time = time.time()
    best_avg_reward = -float("inf")
    for ep in range(1, num_episodes+1):
        total_reward = 0
        obs, _ = env.reset(seed=SEED+ep)

        done, truncated = False, False
        for _ in range(1, EPISODE_MAX_STEP+1):
            action = agent.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            agent.memory.add((obs, action, reward, next_obs, float(not done)))
            obs = next_obs
            agent.train()

            if done or truncated:
                break

        history.append(total_reward)

        # Every 50 episodes: print & save
        if ep % 50 == 0:
            avg_reward = np.mean(history[-50:])
            elapsed_time = time.time() - start_time
            print(f"[Episode {ep}] AvgReward: {avg_reward:.2f} | Time Elapsed: {elapsed_time:.2f}s")

            # Save checkpoint
            torch.save(agent.actor.state_dict(), os.path.join(checkpoint_dir, f"actor_ep{ep}.pth"))
            torch.save(agent.critic1.state_dict(), os.path.join(checkpoint_dir, f"critic1_ep{ep}.pth"))
            torch.save(agent.critic2.state_dict(), os.path.join(checkpoint_dir, f"critic2_ep{ep}.pth"))

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.actor.state_dict(), os.path.join(checkpoint_dir, "best_actor.pth"))
                print(f"✅ New best model saved with AvgReward: {avg_reward:.2f}")
            
            # Plot learning curve
            plot_rewards(history, title="SAC Training Rewards")

if __name__=='__main__':
    checkpoint_dir='ckpt'
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(num_episodes=TOTAL_EPISODES, checkpoint_dir=checkpoint_dir)