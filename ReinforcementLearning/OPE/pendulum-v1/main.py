import gymnasium as gym
import os
import sys
sys.path.append(os.getcwd())
# 主文件
from tools import set_seed, save_results, save_agent
from run import (DiscreteOPE, ContinuousOPE, ReplayBuffer, device, global_seed, Path, run, Env,
                train, evaluate, generate_pendulum_data, create_output_dirs, load_data_to_buffer)

def main():    
    # 1. 初始化环境
    set_seed(global_seed)
    env = gym.make(Env.name)
    state_dim = env.observation_space.shape[0]
   
    # 2. 识别空间并初始化组件
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        agent = DiscreteOPE(state_dim, action_dim, device, run.train.lr, run.train.gamma)
        buffer_action_dim = 1
    else:
        action_dim = env.action_space.shape[0]
        agent = ContinuousOPE(state_dim, action_dim, device, run.train.lr, run.train.gamma)
        buffer_action_dim = action_dim

     # 生成目录
    create_output_dirs()

    # 生成数据, 读取数据
    generate_pendulum_data(episodes=run.generate.epochs, save_path="train_data.csv")
    buffer = ReplayBuffer(state_dim, buffer_action_dim, capacity=100000)
    load_data_to_buffer("train_data.csv", buffer)


    env = gym.make("Pendulum-v1")
    
    agent, train_results = train(agent=agent, buffer=buffer, env=env, )
    save_agent(agent, Path.model.agent)
    save_results(train_results, Path.data.final, "train_results")

    eval_rewards = evaluate(env, agent, epochs=10)
    save_results(eval_rewards, Path.data.final ,"eval_results")

if __name__ == "__main__":
    main()
