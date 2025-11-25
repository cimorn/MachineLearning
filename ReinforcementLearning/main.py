import gym
import numpy as np
import torch
import random
import os
from agent import DQN, DDQN  # 引用新的混合文件
from replay import ReplayBuffer
from runner import Trainer, Evaluator
from config import Config
from plotter import plot_learning_curve, save_animation, plot_return_comparison, plot_eval_comparison

# 兼容性设置
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def set_seed(env, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_experiment(algo, env_name):
    """
    运行单个实验并返回训练和评估数据
    :return: (train_returns, eval_rewards)
    """
    algo_name = algo.__name__
    print(f"\n{'='*10} 正在运行算法: {algo_name} | 环境: {env_name} {'='*10}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 加载配置
    cfg = Config(env_name)
    
    # 2. 初始化环境
    env = gym.make(env_name, render_mode="rgb_array") 
    set_seed(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3. 初始化组件
    replay_buffer = ReplayBuffer(cfg.buffer_size)

    agent = algo(
        state_dim=state_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=action_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon_start,
        target_update=cfg.target_update,
        device=cfg.device
    )

    # 4. 开始训练
    trainer = Trainer(agent, env, replay_buffer, cfg)
    train_returns = trainer.train()

    # 5. 保存单次训练曲线
    plot_path = os.path.join(results_dir, f"{env_name}_{algo_name}_learning_curve.png")
    plot_learning_curve(train_returns, env_name, algo_name, save_path=plot_path)

    # 6. 评估 (运行 5 个 episode 计算平均分)
    eval_env = gym.make(env_name, render_mode="rgb_array") 
    evaluator = Evaluator(agent, eval_env, cfg)
    eval_rewards = evaluator.eval(episodes=5)

    # 录制动画 (仅保存一份 GIF 作为演示)
    print(f"正在录制 {algo_name} 在 {env_name} 上的表现...")
    frames = evaluator.render_episode()
    gif_path = os.path.join(results_dir, f"{env_name}_{algo_name}.gif")
    save_animation(frames, gif_path)
    
    env.close()
    eval_env.close()
    
    return train_returns, eval_rewards

if __name__ == "__main__":
    # 定义要运行的实验
    env_list = ["CartPole-v1", "MountainCar-v0"]  # 如需跑 MountainCar，可添加 "MountainCar-v0"
    algo_list = [DQN, DDQN]

    results_dir = "results"

    for env_name in env_list:
        # 用于存储该环境下所有算法的结果
        all_train_returns = {}
        all_eval_rewards = {}

        for algo in algo_list:
            # 运行实验并收集数据
            t_ret, e_ret = run_experiment(algo, env_name)
            all_train_returns[algo.__name__] = t_ret
            all_eval_rewards[algo.__name__] = e_ret
        
        # --- 绘制对比图 ---
        print(f"\n正在生成 {env_name} 的对比图表...")
        
        # 1. 训练曲线对比
        comp_plot_path = os.path.join(results_dir, f"{env_name}_Comparison_Curve.png")
        plot_return_comparison(all_train_returns, env_name, save_path=comp_plot_path)
        
        # 2. 评估结果对比 (柱状图)
        eval_plot_path = os.path.join(results_dir, f"{env_name}_Evaluation_Comparison.png")
        plot_eval_comparison(all_eval_rewards, env_name, save_path=eval_plot_path)

    print(f"\n所有实验结束。请查看 {results_dir} 文件夹下的对比结果。")