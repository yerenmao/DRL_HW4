import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(scores, title="Training Rewards", save_path="reward.png", step=10):
    scores = np.array(scores)
    averaged_scores = [np.mean(scores[i:i+step]) for i in range(0, len(scores), step)]
    episodes = list(range(step, step * len(averaged_scores) + 1, step))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, averaged_scores, marker='o', label=f"Avg Reward per {step} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
