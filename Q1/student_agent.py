import gymnasium as gym
import numpy as np

import torch
from core.network import Actor
from core.config import STATE_DIM, ACTION_DIM

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        # self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Actor Network
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        # Load the model
        state_dict = torch.load("ckpt/best_actor.pth", map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()


    def act(self, observation):
        # return self.action_space.sample()
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.actor(x)
        action = mean.cpu().numpy()
        return np.clip(action, -2.0, 2.0)
        