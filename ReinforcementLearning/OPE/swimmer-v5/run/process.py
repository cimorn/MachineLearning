import os
import numpy as np
import sys
import pandas as pd
sys.path.append(os.getcwd())
from tools.seed import set_seed
from run.config import global_seed, run, Env

def train(agent, buffer, env):
    set_seed(global_seed*100)
    results = []
    for epoch in range(run.train.epochs):
        epoch_q_losses, epoch_p_losses = [], []
        for _ in range(Env.steps):
            losses = agent.train_step(buffer, run.train.batch)
            
            if isinstance(losses, tuple):
                epoch_q_losses.append(losses[0])
                epoch_p_losses.append(losses[1])
            else:
                epoch_q_losses.append(losses)
                epoch_p_losses.append(0.0)

        if (epoch + 1) % run.train.update == 0: agent.update_target()

        if (epoch + 1) % run.train.eval == 0 or epoch == 0:
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
    return agent, results

def evaluate(env, agent, epochs=run.eval.epochs):
    """在线评估函数"""
    set_seed(global_seed*101)
    rewards = []
    for _ in range(epochs):
        s, _ = env.reset() 
        done = False
        ep_reward = 0
        while not done:
            a = agent.get_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            
            s_next = np.array(s_next, dtype=np.float32).flatten()[:Env.dim.state]
            ep_reward += r
            s = s_next
            done = terminated or truncated

        rewards.append({
            "rewards": ep_reward
        })
    return rewards
