import numpy as np
import torch
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.getcwd())
from run.config import device, Path  # 直接引用你写的工具

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=10000):
        self.device = device
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32) 
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, s, a, r, s_, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device)
        )

    def __len__(self):
        """返回当前可用的经验数量"""
        return self.size

    def clear(self):
        """清空回放池"""
        self.ptr = 0
        self.size = 0

def load_data_to_buffer(csv_name: str, buffer: ReplayBuffer) -> None:
    """
    从CSV读取Pendulum数据并存入ReplayBuffer
    :param csv_path: CSV文件路径
    :param buffer: 初始化的ReplayBuffer实例
    """
    # 读取CSV文件
    df = pd.read_csv(os.path.join(Path.data.raw, csv_name))
    print(f"读取CSV文件: {csv_name} | 总数据量: {len(df)}条")
    
    # 提取数据列并转换为numpy数组
    # 状态（3列）
    states = df[["s1", "s2", "s3"]].values.astype(np.float32)
    # 动作（1列）
    actions = df[["action"]].values.astype(np.float32)
    # 奖励（1列）
    rewards = df[["reward"]].values.astype(np.float32)
    # 下一个状态（3列）
    next_states = df[["ns1", "ns2", "ns3"]].values.astype(np.float32)
    # 终止标志（1列）
    dones = df[["done"]].values.astype(np.float32)
    
    # 清空回放池（防止已有数据）
    buffer.clear()
    
    # 逐行添加到回放池
    print("开始将数据存入ReplayBuffer...")
    for i in range(len(df)):
        buffer.add(
            s=states[i],
            a=actions[i],
            r=rewards[i],
            s_=next_states[i],
            d=dones[i]
        )
        
        # 每1000步打印进度
        if (i+1) % 1000 == 0:
            print(f"已添加 {i+1}/{len(df)} 条数据")
    
    print(f"\n数据存入完成！回放池当前大小: {len(buffer)} | 容量: {buffer.capacity}")