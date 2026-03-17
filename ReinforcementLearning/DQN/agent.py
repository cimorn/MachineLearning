import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ----------------------------------------
# 1. Q网络定义 (Qnet) - 保持不变
# ----------------------------------------
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        # 简单的多层感知机
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) 
        # 注意：输出层不要加激活函数，因为Q值可能是负数（如MountainCar）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------------------------
# 2. DQN 智能体定义 (Agent)
# ----------------------------------------
class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma                  # 折扣因子
        self.epsilon = epsilon              # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0                      # 更新计数器
        self.device = device                # 设备 (CPU/GPU)

        # 初始化 Q 网络 (策略网络) 和 目标网络
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始时将目标网络参数同步为策略网络参数
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # 目标网络不需要计算梯度

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state, is_training=True):
        """
        根据当前状态选择动作
        :param state: 当前状态
        :param is_training: 是否处于训练模式 (训练模式下使用 epsilon-greedy)
        """
        # 训练模式下，以 epsilon 概率随机探索
        if is_training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # 否则（评估模式或 epsilon 概率外），选择 Q 值最大的动作
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state).argmax().item()
            return action

    def update(self, transition_dict):
        """
        核心训练函数：计算 Loss 并更新网络
        """
        # 从字典中取出数据并转为 Tensor
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 1. 计算当前状态的 Q 值: Q(s, a)
        # gather(1, actions) 表示在维度1上根据 actions 索引取值
        q_values = self.q_net(states).gather(1, actions) 

        # 2. 计算目标 Q 值: TD Target
        # max(1)[0] 返回最大值，max(1)[1] 返回索引
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 3. 计算均方误差损失
        loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 4. 反向传播与梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. 定期更新目标网络
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# ----------------------------------------
# 2. DDQN 智能体定义 (Agent)
# ----------------------------------------
class DDQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma                  # 折扣因子
        self.epsilon = epsilon              # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0                      # 更新计数器
        self.device = device                # 设备 (CPU/GPU)

        # 初始化 Q 网络 (策略网络) 和 目标网络
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        
        # 初始时将目标网络参数同步为策略网络参数
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # 目标网络不需要计算梯度

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state, is_training=True):
        """
        根据当前状态选择动作
        :param state: 当前状态
        :param is_training: 是否处于训练模式 (训练模式下使用 epsilon-greedy)
        """
        # 训练模式下，以 epsilon 概率随机探索
        if is_training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # 否则（评估模式或 epsilon 概率外），选择 Q 值最大的动作
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state).argmax().item()
            return action

    def update(self, transition_dict):
        """
        核心训练函数：计算 Loss 并更新网络
        此处已修改为 Double DQN 逻辑
        """
        # 从字典中取出数据并转为 Tensor
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 1. 计算当前状态的 Q 值: Q(s, a)
        # gather(1, actions) 表示在维度1上根据 actions 索引取值
        q_values = self.q_net(states).gather(1, actions) 

        # 2. 计算目标 Q 值: TD Target (DDQN 核心修改部分)
        with torch.no_grad():
            # ====================================================
            # Double DQN Logic
            # ====================================================
            
            # 步骤 A: 使用当前网络 (q_net) 选择下一状态的最优动作
            # argmax(1) 返回最大值的索引，即动作
            max_next_action = self.q_net(next_states).argmax(dim=1, keepdim=True)
            
            # 步骤 B: 使用目标网络 (target_net) 评估该动作的 Q 值
            # 此时不用 .max()，而是用 gather 取出上面选定的那个动作对应的 Q 值
            max_next_q_values = self.target_net(next_states).gather(1, max_next_action)
            
            # ====================================================

            # 计算 TD 目标：R + gamma * Q_target(s', argmax Q_local(s', a))
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # 3. 计算均方误差损失
        loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 4. 反向传播与梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. 定期更新目标网络
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1