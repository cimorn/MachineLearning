import collections
import random
import numpy as np

class ReplayBuffer:
    """
    经验回放池，用于存储交互数据 (state, action, reward, next_state, done)
    打破数据相关性，提高训练稳定性。
    """
    def __init__(self, capacity):
        # 使用 deque 作为队列，先进先出，自动维持最大长度
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):
        """随机采样一个批次"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self):
        """当前缓冲区大小"""
        return len(self.buffer)