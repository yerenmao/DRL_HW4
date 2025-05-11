import random
from collections import deque

import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, limit, device):
        self.buffer = deque(maxlen=limit)
        self.device = device

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini = random.sample(self.buffer, n)
        s, a, r, s2, d = zip(*mini)
        s_batch  = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        a_batch  = torch.tensor(np.array(a), dtype=torch.float32).to(self.device)
        r_batch  = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(-1).to(self.device)
        s2_batch = torch.tensor(np.array(s2), dtype=torch.float32).to(self.device)
        d_batch  = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(-1).to(self.device)
        return s_batch, a_batch, r_batch, s2_batch, d_batch

    def __len__(self):
        return len(self.buffer)