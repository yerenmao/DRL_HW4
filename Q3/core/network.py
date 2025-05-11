import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.config import ACTOR_LR, CRITIC_LR, HIDDEN_LAYER_SIZE, LOG_STD_MIN, LOG_STD_MAX

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), nn.ReLU()
        )
        self.mu_head = nn.Linear(HIDDEN_LAYER_SIZE, action_dim)
        self.log_std_head = nn.Linear(HIDDEN_LAYER_SIZE, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=ACTOR_LR)

    def forward(self, x):
        features = self.net(x)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYER_SIZE),
            nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, HIDDEN_LAYER_SIZE),
            nn.ReLU()
        )
        self.output_net = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_SIZE * 2, HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=CRITIC_LR)

    def forward(self, state, action):
        s_feat = self.state_net(state)
        a_feat = self.action_net(action)
        h = torch.cat([s_feat, a_feat], dim=-1)
        return self.output_net(h)
