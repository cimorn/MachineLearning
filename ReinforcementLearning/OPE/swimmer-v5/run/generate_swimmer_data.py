import gymnasium as gym
import numpy as np
import pandas as pd
import sys
from typing import List, Dict
import os
sys.path.append(os.getcwd())
from tools.seed import set_seed
from run.config import Path, global_seed, Env

def generate_swimmer_data(episodes: int = 10, save_path: str = "swimmer_data.csv") -> None:
    set_seed(global_seed)
    env = gym.make("Swimmer-v5", render_mode=None)
    all_data: List[Dict[str, np.ndarray]] = []
    
    print(f"开始生成{episodes}轮Swimmer数据...")
    for episode in range(episodes):
        state, _ = env.reset(seed=episode)
        episode_reward = 0.0
        
        for step in range(Env.steps):
            action = np.random.uniform(-1.0, 1.0, size=Env.dim.action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            all_data.append({
                "state": state.flatten(),
                "action": action.flatten(),
                "reward": np.array([reward], dtype=np.float32),
                "next_state": next_state.flatten(),
                "done": np.array([1.0 if done else 0.0], dtype=np.float32)
            })
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"第{episode+1}轮完成 | 总奖励: {episode_reward:.2f} | 步数: {step+1}")
    
    env.close()
    
    state_dim = Env.dim.state
    action_dim = Env.dim.action
    
    state_columns = [f"s{i+1}" for i in range(state_dim)]
    action_columns = [f"a{i+1}" for i in range(action_dim)]
    next_state_columns = [f"ns{i+1}" for i in range(state_dim)]
    
    df = pd.DataFrame({
        **{col: [d["state"][i] for d in all_data] for i, col in enumerate(state_columns)},
        **{col: [d["action"][i] for d in all_data] for i, col in enumerate(action_columns)},
        "reward": [d["reward"][0] for d in all_data],
        **{col: [d["next_state"][i] for d in all_data] for i, col in enumerate(next_state_columns)},
        "done": [d["done"][0] for d in all_data]
    })
    
    save_path = os.path.join(Path.data.raw, save_path)
    df.to_csv(save_path, index=False)
    print(f"\n数据已保存至: {save_path} | 总数据量: {len(df)}条")
