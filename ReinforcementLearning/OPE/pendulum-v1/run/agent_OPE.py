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
            nn.Linear(state_dim + action_dim, hidden_dim), # 注意：连续版本通常拼接s和a
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, state, action):
        # 针对连续动作：输入s和a，输出标量Q值
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class DiscreteQNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DiscreteQNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # 输出层：每个动作的Q值
        )

    def forward(self, state):
        return self.model(state)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        # 将上下界转换为张量，方便后续计算（支持单值/多维度动作）

        action_low, action_high=Env.action[0], Env.action[1] 

        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        
        # 确保上下界维度匹配动作维度
        if self.action_low.ndim == 0:  # 如果是单值，扩展为动作维度的张量
            self.action_low = self.action_low.expand(action_dim)
            self.action_high = self.action_high.expand(action_dim)
        
        # 策略网络主干（输出仍为[-1,1]，后续做线性变换）
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 先映射到 [-1, 1]
        )

    def forward(self, s):
        # 先得到[-1,1]的输出
        x = self.fc(s)
        # 确保上下界和输入张量在同一设备（CPU/GPU）
        self.action_low = self.action_low.to(s.device)
        self.action_high = self.action_high.to(s.device)
        # 线性变换到指定区间 [action_low, action_high]
        action = self.action_low + (self.action_high - self.action_low) * (x + 1.0) / 2.0
        return action






class DiscreteOPE:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.99):
        self.device = device
        self.gamma = gamma
        self.q_net = DiscreteQNet(state_dim, action_dim).to(device)
        self.target_net = DiscreteQNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state)
        self.q_net.train()
        return q_values.argmax(dim=1).item()

    def train_step(self, buffer, batch_size):
        s, a, r, s_, d = buffer.sample(batch_size)
        current_q = self.q_net(s).gather(1, a.long()) # 离散动作索引须为long
        with torch.no_grad():
            max_next_q = self.target_net(s_).max(1, keepdim=True)[0]
            target_q = r + (1 - d) * self.gamma * max_next_q
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

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

        # A. 更新Q网络
        with torch.no_grad():
            a_next = self.policy_net(s_) 
            target_q = r + (1 - d) * self.gamma * self.q_target(s_, a_next)
        current_q = self.q_net(s, a)
        q_loss = nn.MSELoss()(current_q, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # B. 更新策略网络 (梯度上升最大化Q值)
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