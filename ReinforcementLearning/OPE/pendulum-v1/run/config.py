from easydict import EasyDict
import torch

global_seed = 42  # 全局随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 环境配置 (Env) =====================
# 包含状态空间、动作空间的范围以及维度定义
Env = EasyDict({
    "name": "Pendulum-v1",
    "state": {
        "cos_theta": (-1.0, 1.0), 
        "sin_theta": (-1.0, 1.0), 
        "theta_dot": (-8.0, 8.0)
    },
    "action": (-2.0, 2.0), # 动作范围：用于后期映射 Tanh 的输出
    "dim": {
        "state": 3, 
        "action": 1, 
        "reward": 1, 
        "nextstate": 3, 
        "Q": 1
    },
    "reward": (-16.2736, 0),
    "steps":200
})

# ===================== 路径配置 (Path) =====================
# 统一管理数据存放、模型权重、结果图表以及日志的路径
Path = EasyDict({
    "data": {
        "raw": "output/data/raw",  
        "final": "output/data/final"
    },
    "model": {
        "agent":"output/model/agent"
    },
    "results": {
        "figures": "output/results/figures", 
        "table": "output/results/table"
    }
})

# 包含模型维度信息、训练超参数以及评估设置
run = EasyDict({
    "generate":{
        "epochs":150
    },
    "dim": {
        "state": Env.dim.state, 
        "action": Env.dim.action, 
        "Q": Env.dim.Q
    },
    "train": {
        "gamma": 0.99, 
        "lr": 3e-4, 
        "epochs": 200,      # 总训练轮数
        "update": 5,        # 目标网络硬更新频率 (每隔多少 epoch)
        "grad_steps": 1,    # 每个 batch 训练时，Q 梯度更新的步数
        "grad_lr": 1e-2,    # 策略寻优时的学习率 (如果涉及内部微调)
        "batch": 100,       # 采样大小
        "eval": 10          # 评估频率 (每隔多少 epoch)
    },
    "eval": {
        "epochs": 100       # 最终离线评估时的轮数
    }
})