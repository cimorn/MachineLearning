import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.getcwd())
from run.config import Env 

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        action_low, action_high = Env.action[0], Env.action[1]

        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        
        if self.action_low.ndim == 0:
            self.action_low = self.action_low.expand(action_dim)
            self.action_high = self.action_high.expand(action_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, s):
        x = self.fc(s)
        self.action_low = self.action_low.to(s.device)
        self.action_high = self.action_high.to(s.device)
        action = self.action_low + (self.action_high - self.action_low) * (x + 1.0) / 2.0
        return action

class ContinuousOPE:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.99):
        self.device = device
        self.gamma = gamma
        self.q_net = QNet(state_dim, action_dim).to(device)
        self.q_target = QNet(state_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.policy_net = PolicyNet(state_dim, action_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).cpu().numpy().flatten()

    def train_step(self, buffer, batch_size):
        s, a, r, s_, d = buffer.sample(batch_size)

        with torch.no_grad():
            a_next = self.policy_net(s_) 
            target_q = r + (1 - d) * self.gamma * self.q_target(s_, a_next)
        current_q = self.q_net(s, a)
        q_loss = nn.MSELoss()(current_q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        for param in self.q_net.parameters(): param.requires_grad = False
        a_predict = self.policy_net(s)
        policy_loss = -self.q_net(s, a_predict).mean() 
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for param in self.q_net.parameters(): param.requires_grad = True

        return q_loss.item(), policy_loss.item()

    def update_target(self):
        self.q_target.load_state_dict(self.q_net.state_dict())
