
import torch
import torch.nn.functional as F
from core.replay_buffer import ReplayBuffer
from core.network import Actor, Critic
from core.config import STATE_DIM, ACTION_DIM, REPLAY_BUFFER_SIZE, ALPHA_LR, BATCH_SIZE, GAMMA, TARGET_ENTROPY, TAU


class Agent:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Replay buffer
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE, self.device)
        # Initialize networks
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic1 = Critic(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic2 = Critic(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic1_tgt = Critic(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic2_tgt = Critic(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())
        # Entropy coefficient (alpha) and optimizer
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    def act(self, observation):
        x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(x)
        return action.cpu().numpy().reshape(-1)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)

        # ----- Critic Update -----
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            alpha = self.log_alpha.exp()
            q1_t = self.critic1_tgt(s2, a2)
            q2_t = self.critic2_tgt(s2, a2)
            q_t = torch.min(q1_t, q2_t) - alpha * logp2
            target = r + GAMMA * d * q_t

        # Q1 loss
        q1 = self.critic1(s, a)
        loss_q1 = F.mse_loss(q1, target)
        self.critic1.optimizer.zero_grad(); loss_q1.backward(); self.critic1.optimizer.step()

        # Q2 loss
        q2 = self.critic2(s, a)
        loss_q2 = F.mse_loss(q2, target)
        self.critic2.optimizer.zero_grad(); loss_q2.backward(); self.critic2.optimizer.step()

        # ----- Actor Update -----
        a_curr, logp = self.actor.sample(s)
        q1_pi = self.critic1(s, a_curr)
        q2_pi = self.critic2(s, a_curr)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        loss_pi = (alpha.detach() * logp - q_pi).mean()
        self.actor.optimizer.zero_grad(); loss_pi.backward(); self.actor.optimizer.step()

        # ----- Alpha Update -----
        loss_alpha = -(self.log_alpha * (logp + TARGET_ENTROPY).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()

        with torch.no_grad():
            self.log_alpha.clamp_(min=-10.0, max=2.0)

        # ----- Soft Update of Target Networks -----
        with torch.no_grad():
            for p_t, p in zip(self.critic1_tgt.parameters(), self.critic1.parameters()):
                p_t.data.mul_(1 - TAU); p_t.data.add_(TAU * p.data)
            for p_t, p in zip(self.critic2_tgt.parameters(), self.critic2.parameters()):
                p_t.data.mul_(1 - TAU); p_t.data.add_(TAU * p.data)
