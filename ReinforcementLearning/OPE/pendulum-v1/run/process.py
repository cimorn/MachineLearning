import os
import numpy as np
import sys
import pandas as pd
sys.path.append(os.getcwd())
from tools.seed import set_seed  # 直接引用你写的工具
from run.config import global_seed,run, Env  # 直接引用你写的工具

def train(agent, buffer, env):
    set_seed(global_seed*100)
    results = []
    # ================= 外层循环：轮次 (Epochs) =================
    for epoch in range(run.train.epochs):
        epoch_q_losses, epoch_p_losses = [], []
        # ================= 内层循环：步数 (Steps) =================
        for _ in range(Env.steps):
            # 由于前面设了种子，这里 buffer.sample 的索引序列在每次运行时都会是一样的
            losses = agent.train_step(buffer, run.train.batch)
            
            # 处理不同 Agent 返回的 Loss 格式
            if isinstance(losses, tuple): # 连续动作 (Q_loss, P_loss)
                epoch_q_losses.append(losses[0])
                epoch_p_losses.append(losses[1])
            else: # 离散动作 (Q_loss)
                epoch_q_losses.append(losses)
                epoch_p_losses.append(0.0)

        # 1. 定期同步 Target 网络 (对应 update 频率)
        if (epoch + 1) % run.train.update == 0: agent.update_target()

        # 2. 定期评估与记录 (对应 eval 频率)
        if (epoch + 1) % run.train.eval == 0 or epoch == 0:
            # 评估时也传入 SEED，保证环境初始状态（如 Pendulum 的初始角度）是固定的
            eval_rewards = evaluate(env, agent, epochs=1)
            mean_q_loss, mean_p_loss = np.mean(epoch_q_losses), np.mean(epoch_p_losses)
            
            print(f"Epoch: [{epoch+1:03d}/{run.train.epochs}] | "
                  f"Q_Loss: {mean_q_loss:.4f} | P_Loss: {mean_p_loss:.4f}")
            
            results.append({
                "epoch": epoch + 1,
                "q_loss": mean_q_loss,
                "p_loss": mean_p_loss,
                "rewards": eval_rewards
            })
    print("\n")
    # ===================== 保存逻辑 =====================
    return agent,results

def evaluate(env, agent, epochs=run.eval.epochs):
    """在线评估函数"""
    # 锁定环境种子
    set_seed(global_seed*101)
    rewards = []
    for _ in range(epochs):
        s, _ = env.reset() 
        done = False
        ep_reward = 0
        while not done:
            a = agent.get_action(s)  # 传入修复后的state
            s_next, r, terminated, truncated, _ = env.step(a)
            
            # 同样修复next_state格式（防呆）
            s_next = np.array(s_next, dtype=np.float32).flatten()[:3]
            ep_reward += r
            s = s_next
            done = terminated or truncated

        rewards.append({
            "rewards": ep_reward
        })
    return rewards
