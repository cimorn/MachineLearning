import gymnasium as gym
import numpy as np
import pandas as pd
import sys
from typing import List, Dict
import os
sys.path.append(os.getcwd())
from tools.seed import set_seed  # 直接引用你写的工具
from run.config import Path, global_seed, Env

def generate_pendulum_data(episodes: int = 10, save_path: str = "pendulum_data.csv") -> None:
    set_seed(global_seed)
    # 初始化环境
    env = gym.make("Pendulum-v1", render_mode=None)
    # 存储所有数据的列表
    all_data: List[Dict[str, np.ndarray]] = []
    
    print(f"开始生成{episodes}轮Pendulum数据...")
    for episode in range(episodes):
        # 重置环境（固定种子保证可复现）
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0
        
        for step in range(Env.steps):
            # 随机生成动作（也可以替换为你的DDPG模型预测动作）
            action = np.random.uniform(-2.0, 2.0, size=Env.dim.action)  # Pendulum动作范围[-2,2]
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 合并终止标志
            
            # 存储单步数据（展平为1维，方便CSV保存）
            all_data.append({
                "state": state.flatten(),
                "action": action.flatten(),
                "reward": np.array([reward], dtype=np.float32),
                "next_state": next_state.flatten(),
                "done": np.array([1.0 if done else 0.0], dtype=np.float32)
            })
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            # 终止条件（Pendulum默认200步截断，无自然终止）
            if done:
                break
        
        print(f"第{episode+1}轮完成 | 总奖励: {episode_reward:.2f} | 步数: {step+1}")
    
    # 关闭环境
    env.close()
    
    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame({
        # 状态（3列）
        "s1": [d["state"][0] for d in all_data],
        "s2": [d["state"][1] for d in all_data],
        "s3": [d["state"][2] for d in all_data],
        # 动作（1列）
        "action": [d["action"][0] for d in all_data],
        # 奖励（1列）
        "reward": [d["reward"][0] for d in all_data],
        # 下一个状态（3列）
        "ns1": [d["next_state"][0] for d in all_data],
        "ns2": [d["next_state"][1] for d in all_data],
        "ns3": [d["next_state"][2] for d in all_data],
        # 终止标志（1列）
        "done": [d["done"][0] for d in all_data]
    })
    
    save_path=os.path.join(Path.data.raw, save_path)
    df.to_csv(save_path, index=False)
    print(f"\n数据已保存至: {save_path} | 总数据量: {len(df)}条")
