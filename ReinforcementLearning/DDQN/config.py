import torch

class Config:
    def __init__(self, env_name):
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 通用参数
        self.render = False  # 是否在训练时渲染
        
        if env_name == 'CartPole-v1':
            self.lr = 2e-3
            self.hidden_dim = 128
            self.gamma = 0.98
            
            # --- 修改重点 1: 增加初始探索 ---
            # 之前是 0.1，太低了，容易陷入局部最优
            self.epsilon_start = 1.0   
            self.epsilon_end = 0.01
            # 调整衰减速度，确保在前 200-300 回合有足够的探索
            self.epsilon_decay = 0.995 
            
            # --- 修改重点 2: 降低目标网络更新频率 ---
            # 之前是 10，太快了，导致 Target Network 和 Local Network 过于相似，
            # 使得 DDQN 的“解耦”失效。增加到 200 能让目标值更稳定。
            self.target_update = 200
            
            self.buffer_size = 10000
            self.minimal_size = 500
            self.batch_size = 64
            self.num_episodes = 600
            
        elif env_name == 'MountainCar-v0':
            self.lr = 1e-3
            self.hidden_dim = 256
            self.gamma = 0.99
            self.epsilon_start = 1.0
            self.epsilon_end = 0.05
            self.epsilon_decay = 0.99
            self.target_update = 50
            self.buffer_size = 20000
            self.minimal_size = 1000
            self.batch_size = 64
            self.num_episodes = 1000
            
        else:
            raise ValueError("未配置的环境参数")

    def print_config(self):
        print(f"\n[{self.env_name}] 环境配置加载完成")
        print(f"设备: {self.device}")
        print(f"学习率: {self.lr}, 隐藏层: {self.hidden_dim}")
        print(f"Target Update: {self.target_update}") # 打印一下这个参数
        print(f"Epsilon Start: {self.epsilon_start}")
        print(f"总回合数: {self.num_episodes}")