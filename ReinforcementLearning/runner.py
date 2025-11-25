import numpy as np
from tqdm import tqdm
# 给耗时的循环加个 “进度条”

class Trainer:
    def __init__(self, agent, env, replay_buffer, cfg):
        self.agent = agent
        self.env = env
        self.buffer = replay_buffer
        self.cfg = cfg
        
    def train(self):
        """执行完整的训练流程"""
        return_list = [] # 记录每一回合的回报
        
        # 使用 tqdm 显示进度条
        with tqdm(total=int(self.cfg.num_episodes), desc='Training') as pbar:
            for i_episode in range(int(self.cfg.num_episodes)):
                
                # 重置环境
                state, _ = self.env.reset()
                done = False
                episode_return = 0

                while not done:
                    # 1. 智能体选择动作
                    action = self.agent.take_action(state)
                    
                    # 2. 环境执行动作
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # 核心交互：让智能体执行动作action，环境返回5个关键反馈（Gym 0.26+版本标准返回格式）
                    # 返回值说明（按顺序对应）：
                    # next_state: 执行动作后的下一状态（CartPole中为4维向量：小车位置、速度、摆杆角度、角速度）
                    # reward: 环境对当前动作的即时奖励（CartPole中每坚持1步得1.0分，鼓励平衡）
                    # terminated: 任务终止标志（bool）→ True=摆杆倾倒/小车越界（任务本身结束）
                    # truncated: 截断标志（bool）→ True=达到最大步数（外部限制强制结束）
                    # _: 占位符，忽略环境返回的额外辅助信息（无需使用）
                    
                    # MountainCar 特殊处理: 到达山顶(pos >= 0.5)给大奖励，帮助收敛
                    if self.cfg.env_name == 'MountainCar-v0' and next_state[0] >= 0.5:
                        reward += 100

                    # 3. 存入经验池
                    self.buffer.add(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_return += reward

                    # 4. 如果经验池够了，开始训练
                    if self.buffer.size() > self.cfg.minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = self.buffer.sample(self.cfg.batch_size)
                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a, 
                            'next_states': b_ns, 
                            'rewards': b_r, 
                            'dones': b_d
                        }
                        self.agent.update(transition_dict)

                # 更新 epsilon (探索率衰减)
                self.agent.epsilon = max(
                    self.cfg.epsilon_end, 
                    self.agent.epsilon * self.cfg.epsilon_decay
                )

                # 记录数据与更新进度条
                return_list.append(episode_return)
                pbar.set_postfix({
                    'episode': '%d' % (i_episode + 1),
                    'return': '%.2f' % np.mean(return_list[-10:]), # 最近10次平均
                    'eps': '%.2f' % self.agent.epsilon
                })
                pbar.update(1)
        
        return return_list

class Evaluator:
    def __init__(self, agent, env, cfg):
        self.agent = agent
        self.env = env
        self.cfg = cfg
        
    def eval(self, episodes=5):
        """评估模型表现"""
        print(f"\n开始评估 {self.cfg.env_name}，共 {episodes} 回合...")
        total_rewards = []
        
        for i in range(episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                # 评估时 is_training=False，不使用 epsilon-greedy，只选最优
                action = self.agent.take_action(state, is_training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = next_state
            
            total_rewards.append(ep_reward)
            print(f"评估回合 {i+1}/{episodes}: 回报 = {ep_reward:.2f}")
            
        print(f"平均回报: {np.mean(total_rewards):.2f}")
        return total_rewards

    # --- 新增的方法 ---
    def render_episode(self):
        """
        录制一个回合的动画帧
        注意：env 必须以 render_mode='rgb_array' 初始化
        """
        frames = []
        state, _ = self.env.reset()
        done = False
        
        # 记录初始帧
        first_frame = self.env.render()
        if first_frame is not None:
            frames.append(first_frame)
        
        while not done:
            action = self.agent.take_action(state, is_training=False)
            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 获取当前帧画面并加入列表
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)
            
        return frames