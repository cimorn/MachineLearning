import gymnasium as gym
import os
import sys
sys.path.append(os.getcwd())
from tools import set_seed, save_results, save_agent
from run import (ContinuousOPE, ReplayBuffer, device, global_seed, Path, run, Env,
                train, evaluate, generate_swimmer_data, create_output_dirs, load_data_to_buffer, add_noise_to_data)

def main():    
    set_seed(global_seed)
    env = gym.make(Env.name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"环境: {Env.name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
   
    agent = ContinuousOPE(state_dim, action_dim, device, run.train.lr, run.train.gamma)
    buffer_action_dim = action_dim

    create_output_dirs()

    # generate_swimmer_data(episodes=run.generate.epochs, save_path="train_data.csv")
    # add_noise_to_data(csv_name="train_data.csv", output_name="val_data.csv", noise_std=0.1)
    buffer = ReplayBuffer(state_dim, buffer_action_dim, capacity=100000)
    load_data_to_buffer("val_data.csv", buffer)

    agent, train_results = train(agent=agent, buffer=buffer, env=env)
    save_agent(agent, Path.model.agent)
    save_results(train_results, Path.data.final, "train_results")

    eval_rewards = evaluate(env, agent, epochs=run.eval.epochs)
    save_results(eval_rewards, Path.data.final, "eval_results")

if __name__ == "__main__":
    main()
